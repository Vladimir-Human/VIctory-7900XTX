//! Fenced GPU-to-CPU readback helper for card-buffer demotion
//! (`crate::shadow::demote_cold_small_global`).

use std::sync::Mutex;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FENCE_FLAG_NONE, ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue,
    ID3D12Fence, ID3D12GraphicsCommandList, ID3D12Resource,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use super::{Host, Prelude, QUEUE_SUBMIT};

/// Dedicated list, fence, and event for reading back cold card-resident
/// buffers before demoting them to host RAM (`crate::shadow::demote_cold_small_global`).
/// Staging memory is allocated transiently per batch and dropped immediately
/// after readback so idle L0 RAM is never pinned.
struct DemoteCopy {
    list: Prelude,
    fence: ID3D12Fence,
    event: HANDLE,
    sent: u64,
}

impl Drop for DemoteCopy {
    fn drop(&mut self) {
        // SAFETY: `self.event` was created by `CreateEventW` and is owned
        // exclusively by `DemoteCopy`.
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

// SAFETY: D3D12 command allocators, lists, fences, and Win32 events are used
// exclusively under `DEMOTE_COPY`'s mutex, one readback at a time.
unsafe impl Send for DemoteCopy {}

static DEMOTE_COPY: Mutex<Option<DemoteCopy>> = Mutex::new(None);

/// Executes one fenced batch readback of `srcs` into `staging` (persistently
/// mapped at `staging_data`) on `queue`, returning the exact bytes per source.
/// When `queue_lock` is provided, holds it strictly across
/// `ExecuteCommandLists` / `on_sent` / `Signal` (released before the fence wait).
#[allow(clippy::too_many_arguments)]
pub(crate) fn demote_readback_run(
    allocator: &ID3D12CommandAllocator,
    list: &ID3D12GraphicsCommandList,
    queue: &ID3D12CommandQueue,
    fence: &ID3D12Fence,
    event: HANDLE,
    signal: u64,
    staging: &ID3D12Resource,
    staging_data: *const u8,
    srcs: &[(ID3D12Resource, u64)],
    queue_lock: Option<&Mutex<()>>,
    on_sent: Option<&dyn Fn()>,
) -> Option<Vec<Vec<u8>>> {
    // SAFETY: caller guarantees `allocator`/`list` are idle, `staging` holds at
    // least `sum(len)` bytes persistently mapped at `staging_data`, every source
    // buffer stays alive across the fence wait, and `event` is exclusive.
    let ran = unsafe {
        (|| -> windows::core::Result<()> {
            allocator.Reset()?;
            list.Reset(allocator, None)?;
            let mut at = 0u64;
            for (src, len) in srcs {
                if *len > 0 {
                    list.CopyBufferRegion(staging, at, src, 0, *len);
                    at += *len;
                }
            }
            list.Close()?;
            let lists = [Some(list.cast::<ID3D12CommandList>()?)];
            {
                let _q = queue_lock.map(|m| m.lock().unwrap_or_else(|e| e.into_inner()));
                queue.ExecuteCommandLists(&lists);
                if let Some(sent_cb) = on_sent {
                    sent_cb();
                }
                queue.Signal(fence, signal)?;
            }
            fence.SetEventOnCompletion(signal, event)?;
            WaitForSingleObject(event, INFINITE);
            Ok(())
        })()
    };
    if let Err(e) = ran {
        tracing::warn!(error = %e, "card-buffer demotion readback failed; keeping card placements");
        return None;
    }
    let mut out = Vec::with_capacity(srcs.len());
    let mut at = 0usize;
    for &(_, len) in srcs {
        let len = len as usize;
        if len == 0 {
            out.push(Vec::new());
            continue;
        }
        // SAFETY: the GPU copy into `staging[at..at + len]` has completed and
        // `staging` is persistently mapped at `staging_data`.
        let slice = unsafe { std::slice::from_raw_parts(staging_data.add(at), len) };
        out.push(slice.to_vec());
        at += len;
    }
    Some(out)
}

/// Host entry point for [`Host::demote_readback`].
pub(super) fn demote_readback_host(host: &Host, srcs: &[(ID3D12Resource, u64)]) -> Option<Vec<Vec<u8>>> {
    if srcs.is_empty() {
        return Some(Vec::new());
    }
    let total: u64 = srcs.iter().try_fold(0u64, |acc, &(_, len)| acc.checked_add(len))?;
    if total == 0 {
        return Some(srcs.iter().map(|_| Vec::new()).collect());
    }
    // Allocate staging transiently so up to 16 MiB of L0 RAM is released
    // immediately once this demotion batch finishes.
    let (staging, data) = host.shared_buffer(total.next_multiple_of(64 << 10))?;
    let mut slot = DEMOTE_COPY.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        let list = host.make_prelude()?;
        // SAFETY: plain D3D12 fence and Win32 auto-reset event.
        let made = unsafe {
            (|| -> windows::core::Result<DemoteCopy> {
                let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                let event = CreateEventW(None, false, false, None)?;
                Ok(DemoteCopy {
                    list,
                    fence,
                    event,
                    sent: 0,
                })
            })()
        };
        *slot = made.ok();
    }
    let copy = slot.as_mut()?;
    copy.sent += 1;
    let on_sent = || host.card_sent_locked();
    let result = demote_readback_run(
        &copy.list.0,
        &copy.list.1,
        &host.queue,
        &copy.fence,
        copy.event,
        copy.sent,
        &staging,
        data.cast_const(),
        srcs,
        Some(&QUEUE_SUBMIT),
        Some(&on_sent),
    );
    if result.is_none() {
        // Reset `DemoteCopy` (closing its Win32 event via `Drop`) so a
        // failed `Reset`/`Close`/`Signal` never poisons future readbacks.
        *slot = None;
    }
    drop(slot);
    drop(staging);
    result
}
