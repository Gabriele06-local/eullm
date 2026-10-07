//! Reading a model's GGUF ahead of llama.cpp's load, with several threads
//! (`--load-threads`).
//!
//! llama.cpp reads the file it loads in a single stream. From a local disk
//! that is fast enough; from a network file system it is not, because one
//! stream waits on every request it makes, and many do not wait on each
//! other. Measured on a LUMI-G compute node on 06-10-2026, reading a part of
//! a Lustre-stored model through the page cache (`tools/lumi/sbatch_lustre_probe.slurm`):
//! one stream 178 MB/s, four 681 MB/s, sixteen 2,283 MB/s in all. At the
//! single-stream rate the Coder-480B (290 GB) had not finished loading after
//! an hour.
//!
//! So as a load starts, readers go over the model's parts in file order and
//! bring them into the page cache; llama.cpp, a step behind, then reads them
//! from memory. Nothing is kept by the readers themselves: the cache is the
//! kernel's, and its pages are given back as soon as anything needs the
//! memory. That is also the limit of the idea: a model larger than the memory
//! free for it would push its own first pages out before llama.cpp reached
//! them, and be read twice, so it is then read in one stream as before.
//!
//! `auto`, the default, reads ahead only from a network file system (Lustre,
//! NFS, SMB, GPFS, BeeGFS, CephFS, 9p): a local disk is not where the time
//! went, and keeps the load it always had until measured otherwise.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::Instant;

/// Readers for `auto` on a network file system: the point where the LUMI
/// measurement above stopped, and still rising there.
pub const AUTO_THREADS: u32 = 16;
/// Upper bound for an explicit `--load-threads N`.
pub const MAX_THREADS: u32 = 64;
/// Below this a model loads in seconds from anywhere: no readers.
const MIN_BYTES: u64 = 1 << 30;
/// Each reader takes this much of a part at a time, in file order.
const CHUNK: u64 = 64 << 20;
/// The share of the memory free for the file that it may fill.
const MEMORY_SHARE: f64 = 0.9;

/// `--load-threads`: `auto` (network file systems only) or a number of
/// readers, 0 for none.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoadThreads {
    #[default]
    Auto,
    Fixed(u32),
}

impl std::fmt::Display for LoadThreads {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadThreads::Auto => f.write_str("auto"),
            LoadThreads::Fixed(n) => write!(f, "{n}"),
        }
    }
}

/// clap's parser for `--load-threads`.
pub fn parse_load_threads(s: &str) -> Result<LoadThreads, String> {
    if s.eq_ignore_ascii_case("auto") {
        return Ok(LoadThreads::Auto);
    }
    match s.parse::<u32>() {
        Ok(n) if n <= MAX_THREADS => Ok(LoadThreads::Fixed(n)),
        _ => Err(format!(
            "expected auto or a number of threads from 0 to {MAX_THREADS}"
        )),
    }
}

/// How many readers a load gets, and why none when it gets none: the whole
/// decision, without touching the disk, so it can be tested.
///
/// `network` names the file system when it is a network one; `memory` is
/// what the page cache may hold (see [`memory_for_cache`]), `None` where the
/// platform does not say.
pub fn plan(
    setting: LoadThreads,
    network: Option<&str>,
    bytes: u64,
    memory: Option<u64>,
) -> Result<u32, Option<String>> {
    let threads = match setting {
        LoadThreads::Fixed(0) => return Err(None),
        LoadThreads::Fixed(n) => n,
        LoadThreads::Auto if network.is_some() => AUTO_THREADS,
        LoadThreads::Auto => return Err(None),
    };
    if bytes < MIN_BYTES {
        return Err(None);
    }
    if let Some(memory) = memory
        && bytes as f64 > memory as f64 * MEMORY_SHARE
    {
        return Err(Some(format!(
            "--load-threads: not reading ahead: the model ({}) does not fit in the {} of \
             memory free for the page cache, so it would push its own first pages out",
            crate::fit::gib(bytes),
            crate::fit::gib(memory)
        )));
    }
    Ok(threads)
}

/// Readers running over a model's parts. [`ReadAhead::finish`] once the load
/// is over, whichever way it ended.
pub struct ReadAhead {
    stop: Arc<AtomicBool>,
    read: Arc<AtomicU64>,
    readers: Vec<JoinHandle<()>>,
    total: u64,
    started: Instant,
}

/// Start reading `path` (every part of a split model) ahead of its load, or
/// return `None` when the setting, the file system or the memory says not
/// to (logged when it is the memory).
pub fn start(path: &Path, setting: LoadThreads) -> Option<ReadAhead> {
    if setting == LoadThreads::Fixed(0) {
        return None;
    }
    let parts = crate::fit::gguf_parts(path);
    let sizes: Vec<u64> = parts
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .collect();
    let total: u64 = sizes.iter().sum();
    let network = network_file_system(path);
    let threads = match plan(setting, network, total, memory_for_cache()) {
        Ok(n) => n,
        Err(why) => {
            if let Some(why) = why {
                tracing::info!("{why}");
            }
            return None;
        }
    };
    tracing::info!(
        "--load-threads: {threads} threads read the model ({}{}) ahead of the load",
        crate::fit::gib(total),
        network.map(|fs| format!(", on {fs}")).unwrap_or_default()
    );
    let chunks = Arc::new(chunks(&sizes));
    let parts = Arc::new(parts);
    let next = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let read = Arc::new(AtomicU64::new(0));
    let readers = (0..threads)
        .filter_map(|i| {
            let (chunks, parts, next, stop, read) = (
                chunks.clone(),
                parts.clone(),
                next.clone(),
                stop.clone(),
                read.clone(),
            );
            std::thread::Builder::new()
                .name(format!("readahead-{i}"))
                .spawn(move || reader(&parts, &chunks, &next, &stop, &read))
                .ok()
        })
        .collect();
    Some(ReadAhead {
        stop,
        read,
        readers,
        total,
        started: Instant::now(),
    })
}

impl ReadAhead {
    /// Stop the readers (the load no longer needs them, or failed) and log
    /// what they read, at what rate.
    pub fn finish(self) {
        self.stop.store(true, Ordering::Relaxed);
        let threads = self.readers.len();
        for r in self.readers {
            let _ = r.join();
        }
        let read = self.read.load(Ordering::Relaxed);
        let secs = self.started.elapsed().as_secs_f64().max(1e-3);
        tracing::info!(
            "--load-threads: {} of {} read ahead by {threads} threads in {secs:.1} s ({:.0} MB/s)",
            crate::fit::gib(read),
            crate::fit::gib(self.total),
            read as f64 / secs / 1e6
        );
    }
}

/// (part, offset, length) of every chunk, part by part in file order: the
/// order llama.cpp reads the tensors in, so the readers stay ahead of it.
fn chunks(sizes: &[u64]) -> Vec<(usize, u64, u64)> {
    let mut out = Vec::new();
    for (part, &size) in sizes.iter().enumerate() {
        let mut off = 0;
        while off < size {
            let len = CHUNK.min(size - off);
            out.push((part, off, len));
            off += len;
        }
    }
    out
}

fn reader(
    parts: &[PathBuf],
    chunks: &[(usize, u64, u64)],
    next: &AtomicUsize,
    stop: &AtomicBool,
    read: &AtomicU64,
) {
    let mut files: Vec<Option<File>> = parts.iter().map(|_| None).collect();
    let mut buf = vec![0u8; 8 << 20];
    while !stop.load(Ordering::Relaxed) {
        let Some(&(part, off, len)) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) else {
            return;
        };
        let file = match &mut files[part] {
            Some(f) => f,
            slot => match File::open(&parts[part]) {
                Ok(f) => slot.insert(f),
                // llama.cpp opens the same file and reports what is wrong.
                Err(_) => return,
            },
        };
        if file.seek(SeekFrom::Start(off)).is_err() {
            return;
        }
        let mut left = len;
        while left > 0 && !stop.load(Ordering::Relaxed) {
            let want = left.min(buf.len() as u64) as usize;
            match file.read(&mut buf[..want]) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    left -= n as u64;
                    read.fetch_add(n as u64, Ordering::Relaxed);
                }
            }
        }
    }
}

/// The network file system `path` is on, by name, or `None` for a local one
/// (and wherever the platform does not say).
#[cfg(target_os = "linux")]
pub fn network_file_system(path: &Path) -> Option<&'static str> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a valid NUL-terminated path and `st` a properly sized,
    // writable statfs; statfs writes nothing else.
    if unsafe { libc::statfs(c.as_ptr(), &raw mut st) } != 0 {
        return None;
    }
    network_magic(st.f_type as u64)
}

#[cfg(not(target_os = "linux"))]
pub fn network_file_system(_path: &Path) -> Option<&'static str> {
    None
}

/// Network file systems by the magic number Linux's `statfs` reports.
fn network_magic(magic: u64) -> Option<&'static str> {
    Some(match magic {
        0x0BD0_0BD0 => "Lustre",
        0x6969 => "NFS",
        0xFF53_4D42 | 0xFE53_4D42 | 0x517B => "SMB",
        0x4750_4653 => "GPFS",
        0x1983_0326 => "BeeGFS",
        0x00C3_6400 => "CephFS",
        0x0102_1997 => "9p",
        _ => return None,
    })
}

/// What the page cache may hold for a load: the memory the kernel calls
/// available, and no more than this process's cgroup is allowed (a Slurm
/// job's `--mem`), whichever is smaller.
fn memory_for_cache() -> Option<u64> {
    let available = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| meminfo_available(&m));
    let limit = std::fs::read_to_string("/proc/self/cgroup")
        .ok()
        .and_then(|c| cgroup_memory_limit(&c, Path::new("/sys/fs/cgroup")));
    match (available, limit) {
        (Some(a), Some(l)) => Some(a.min(l)),
        (a, l) => a.or(l),
    }
}

fn meminfo_available(meminfo: &str) -> Option<u64> {
    meminfo.lines().find_map(|l| {
        let kb = l
            .strip_prefix("MemAvailable:")?
            .trim()
            .trim_end_matches("kB")
            .trim();
        kb.parse::<u64>().ok().map(|kb| kb * 1024)
    })
}

/// The memory limit of the cgroup `/proc/self/cgroup` names, under `root`:
/// v2's `memory.max`, or v1's `memory.limit_in_bytes`. `None` for no limit.
fn cgroup_memory_limit(proc_self_cgroup: &str, root: &Path) -> Option<u64> {
    for line in proc_self_cgroup.lines() {
        let mut fields = line.splitn(3, ':');
        let (_, controllers, path) = (fields.next()?, fields.next()?, fields.next()?);
        let rel = path.trim_start_matches('/');
        let file = if controllers.is_empty() {
            root.join(rel).join("memory.max")
        } else if controllers.split(',').any(|c| c == "memory") {
            root.join("memory").join(rel).join("memory.limit_in_bytes")
        } else {
            continue;
        };
        // "max" (v2) or a huge page-rounded number (v1) both mean no limit.
        let value = std::fs::read_to_string(file).ok()?;
        return value.trim().parse::<u64>().ok().filter(|&v| v < (1 << 60));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("eullm-{name}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn the_flag_reads_auto_or_a_bounded_number() {
        assert_eq!(parse_load_threads("auto"), Ok(LoadThreads::Auto));
        assert_eq!(parse_load_threads("AUTO"), Ok(LoadThreads::Auto));
        assert_eq!(parse_load_threads("0"), Ok(LoadThreads::Fixed(0)));
        assert_eq!(parse_load_threads("16"), Ok(LoadThreads::Fixed(16)));
        for bad in ["", "-1", "65", "lots"] {
            assert!(parse_load_threads(bad).is_err(), "accepted {bad:?}");
        }
        assert_eq!(LoadThreads::Auto.to_string(), "auto");
        assert_eq!(LoadThreads::Fixed(8).to_string(), "8");
    }

    #[test]
    fn auto_reads_ahead_only_from_a_network_file_system() {
        let big = 290 * GIB;
        let ram = Some(450 * GIB);
        assert_eq!(
            plan(LoadThreads::Auto, Some("Lustre"), big, ram),
            Ok(AUTO_THREADS)
        );
        assert_eq!(plan(LoadThreads::Auto, None, big, ram), Err(None));
        assert_eq!(plan(LoadThreads::Fixed(4), None, big, ram), Ok(4));
        assert_eq!(
            plan(LoadThreads::Fixed(0), Some("Lustre"), big, ram),
            Err(None)
        );
    }

    #[test]
    fn a_small_model_or_one_larger_than_the_cache_is_read_in_one_stream() {
        assert_eq!(plan(LoadThreads::Fixed(8), None, GIB / 2, None), Err(None));
        // DeepSeek-V3.1 Q4_K_M in a 60 GB job: it would evict its own pages.
        let why = plan(LoadThreads::Auto, Some("Lustre"), 378 * GIB, Some(60 * GIB))
            .unwrap_err()
            .unwrap();
        assert!(why.contains("push its own first pages out"), "{why}");
        // Unknown memory: read ahead, the kernel still reclaims.
        assert_eq!(
            plan(LoadThreads::Auto, Some("NFS"), 378 * GIB, None),
            Ok(AUTO_THREADS)
        );
    }

    #[test]
    fn chunks_cover_every_part_in_file_order() {
        let sizes = [CHUNK * 2 + 5, 0, CHUNK - 1];
        let c = chunks(&sizes);
        assert_eq!(
            c,
            vec![
                (0, 0, CHUNK),
                (0, CHUNK, CHUNK),
                (0, 2 * CHUNK, 5),
                (2, 0, CHUNK - 1)
            ]
        );
        let per_part = |p| c.iter().filter(|x| x.0 == p).map(|x| x.2).sum::<u64>();
        assert_eq!(per_part(0), sizes[0]);
        assert_eq!(per_part(2), sizes[2]);
    }

    #[test]
    fn network_file_systems_are_known_by_their_magic() {
        assert_eq!(network_magic(0x0BD0_0BD0), Some("Lustre"));
        assert_eq!(network_magic(0x6969), Some("NFS"));
        assert_eq!(network_magic(0xEF53), None); // ext4
        assert_eq!(network_magic(0x5846_5342), None); // xfs
    }

    #[test]
    fn the_cache_is_bounded_by_meminfo_and_the_jobs_cgroup() {
        assert_eq!(
            meminfo_available("MemTotal: 100 kB\nMemAvailable:   2048 kB\n"),
            Some(2 << 20)
        );
        let root = scratch_dir("cgroup");
        let job = root.join("system.slice/slurmstepd.scope/job_1");
        std::fs::create_dir_all(&job).unwrap();
        std::fs::write(job.join("memory.max"), "64424509440\n").unwrap();
        let v2 = "0::/system.slice/slurmstepd.scope/job_1\n";
        assert_eq!(cgroup_memory_limit(v2, &root), Some(60 * GIB));
        std::fs::write(job.join("memory.max"), "max\n").unwrap();
        assert_eq!(cgroup_memory_limit(v2, &root), None);

        let v1dir = root.join("memory/slurm/job_2");
        std::fs::create_dir_all(&v1dir).unwrap();
        std::fs::write(v1dir.join("memory.limit_in_bytes"), "8589934592\n").unwrap();
        let v1 = "12:pids:/slurm/job_2\n4:memory:/slurm/job_2\n";
        assert_eq!(cgroup_memory_limit(v1, &root), Some(8 * GIB));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_readers_read_every_byte_of_every_part() {
        let dir = scratch_dir("readahead");
        let a = dir.join("m-00001-of-00002.gguf");
        let b = dir.join("m-00002-of-00002.gguf");
        std::fs::write(&a, vec![1u8; (CHUNK + 3) as usize]).unwrap();
        std::fs::write(&b, vec![2u8; 1000]).unwrap();
        let sizes = [CHUNK + 3, 1000];
        let (chunks, next, stop, read) = (
            chunks(&sizes),
            AtomicUsize::new(0),
            AtomicBool::new(false),
            AtomicU64::new(0),
        );
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| reader(&[a.clone(), b.clone()], &chunks, &next, &stop, &read));
            }
        });
        assert_eq!(read.load(Ordering::Relaxed), CHUNK + 3 + 1000);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
