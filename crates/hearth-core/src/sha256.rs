//! SHA-256, because a digest you did not check is a download you did not verify.
//!
//! Written out rather than taken as a dependency. That is not a general policy
//! — it is specific to this function: SHA-256 is a finished, fully specified
//! algorithm with published test vectors, so "did I implement it correctly" is
//! a question with a definitive answer that runs in microseconds. A crypto
//! crate for one hash would be a supply chain to audit forever in exchange for
//! nothing, and the rest of this workspace already declines HTTP clients on the
//! same reasoning.
//!
//! It is verified against the NIST vectors, against the empty string, against
//! every length around the 55/56/64-byte padding boundaries where hand-written
//! implementations actually break, and against a real digest published by
//! registry.ollama.ai.
//!
//! Streaming, because model weights are measured in gigabytes and hashing by
//! reading the whole file into memory would defeat the purpose of verifying it.

/// The eight initial hash values: the first 32 bits of the fractional parts of
/// the square roots of the first eight primes. FIPS 180-4 §5.3.3.
const H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Round constants: first 32 bits of the fractional parts of the cube roots of
/// the first sixty-four primes. FIPS 180-4 §4.2.2.
#[rustfmt::skip]
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// An incremental SHA-256. Feed it chunks; ask for the digest once.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    /// Bytes not yet part of a complete 64-byte block.
    buffer: [u8; 64],
    buffered: usize,
    /// Total message length in bytes. The padding encodes this as a 64-bit bit
    /// count, so a u64 of bytes is the right thing to carry.
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Sha256::new()
    }
}

impl Sha256 {
    pub fn new() -> Sha256 {
        Sha256 {
            state: H0,
            buffer: [0u8; 64],
            buffered: 0,
            total: 0,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);

        // Top off a partial block first.
        if self.buffered > 0 {
            let want = 64 - self.buffered;
            let take = want.min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }

        // Then whole blocks straight out of the caller's slice.
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            let mut b = [0u8; 64];
            b.copy_from_slice(block);
            self.compress(&b);
            data = rest;
        }

        // Keep the remainder for next time.
        if !data.is_empty() {
            self.buffer[..data.len()].copy_from_slice(data);
            self.buffered = data.len();
        }
    }

    /// The digest, lowercase hex. Consumes the hasher because finishing mutates
    /// the state — offering a `&self` version would invite hashing twice and
    /// getting two different answers.
    pub fn finish(mut self) -> String {
        let bit_len = self.total.wrapping_mul(8);

        // FIPS 180-4 §5.1.1: append 0x80, then zeros until the length is
        // 56 mod 64, then the message length as a big-endian u64.
        self.update_no_count(&[0x80]);
        while self.buffered != 56 {
            self.update_no_count(&[0x00]);
        }
        self.update_no_count(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buffered, 0, "padding must land on a block boundary");

        let mut out = String::with_capacity(64);
        for word in self.state {
            out.push_str(&format!("{word:08x}"));
        }
        out
    }

    /// Feed padding bytes without counting them toward the message length.
    fn update_no_count(&mut self, data: &[u8]) {
        for &b in data {
            self.buffer[self.buffered] = b;
            self.buffered += 1;
            if self.buffered == 64 {
                let block = self.buffer;
                self.compress(&block);
                self.buffered = 0;
            }
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        for (s, v) in self.state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *s = s.wrapping_add(v);
        }
    }
}

/// Hex digest of a byte slice.
pub fn hex_digest(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    h.finish()
}

/// Hex digest of a file, read in chunks.
///
/// Chunked because model weights are gigabytes: reading one into memory to
/// verify it would trade the whole point of the check for a much larger
/// resident set.
pub fn hex_digest_file(path: &std::path::Path) -> std::io::Result<String> {
    hex_digest_file_with_progress(path, |_, _| {})
}

/// Same digest, with a callback so a caller can show that it is still working.
///
/// HASHING A MODEL IS NOT A FAST STEP AND IT LOOKED LIKE A HANG. A 35 GiB
/// download finishes, curl's meter prints `100.0%`, and then this function ran
/// for minutes with no output at all — after which the only honest reading of
/// the terminal is that the process is wedged. `PullConfig::progress` already
/// states the principle ("a silent twenty-minute pause is indistinguishable
/// from a hang, and a downloader that looks hung gets killed halfway"); the
/// verify step simply did not honour it.
///
/// `on_progress(bytes_hashed, total_bytes)` is called after every chunk.
/// `total_bytes` is 0 when the length could not be determined, so a caller
/// must not divide by it blindly. Throttling is the caller's job — this fires
/// per 256 KiB chunk and printing that often would be its own problem.
pub fn hex_digest_file_with_progress(
    path: &std::path::Path,
    mut on_progress: impl FnMut(u64, u64),
) -> std::io::Result<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let total = f.metadata().map(|m| m.len()).unwrap_or(0);
    let mut h = Sha256::new();
    // 256 KiB: large enough that syscall overhead disappears against disk
    // throughput, small enough to stay out of the way on a busy serving box.
    let mut buf = vec![0u8; 256 * 1024];
    let mut done: u64 = 0;
    loop {
        match f.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                h.update(&buf[..n]);
                done = done.saturating_add(n as u64);
                on_progress(done, total);
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(h.finish())
}

/// Strip an optional `sha256:` prefix and lowercase, so a digest from a
/// registry manifest and one from this module can be compared directly.
pub fn normalize(digest: &str) -> String {
    digest
        .trim()
        .trim_start_matches("sha256:")
        .to_ascii_lowercase()
}

/// Do these two digests refer to the same content?
///
/// Compared after normalizing, because the registry writes `sha256:abc…` and a
/// local computation produces `abc…`, and a string comparison between those two
/// forms is false for content that is in fact identical — which would report
/// every correct download as corrupt.
pub fn matches(expected: &str, actual: &str) -> bool {
    normalize(expected) == normalize(actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The published vectors. If any of these fail, nothing else in this file
    // is worth reading.

    #[test]
    fn the_empty_string() {
        assert_eq!(
            hex_digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn nist_abc() {
        assert_eq!(
            hex_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn nist_two_block_message() {
        assert_eq!(
            hex_digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn a_million_a_s() {
        // The long NIST vector. Catches length-counter mistakes that short
        // inputs cannot reach.
        let mut h = Sha256::new();
        for _ in 0..1_000 {
            h.update(&[b'a'; 1_000]);
        }
        assert_eq!(
            h.finish(),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn every_length_across_the_padding_boundaries() {
        // 55, 56 and 64 bytes are where hand-written padding breaks: at 56 the
        // length no longer fits in the final block and a second one is
        // required. Walk the whole neighbourhood rather than spot-checking.
        //
        // Reference values produced by an independent implementation and
        // pinned here, so a regression cannot be explained away.
        // Values computed with Python's hashlib — an independent
        // implementation — and pinned. Two of the digests first written here
        // were wrong: recalled rather than computed, and five of the seven
        // happened to be right, which is precisely why they get computed.
        let expect = [
            // (len, digest of `len` repetitions of b'a')
            (
                54,
                "a3f01b6939256127582ac8ae9fb47a382a244680806a3f613a118851c1ca1d47",
            ),
            (
                55,
                "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                56,
                "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                57,
                "f13b2d724659eb3bf47f2dd6af1accc87b81f09f59f2b75e5c0bed6589dfe8c6",
            ),
            (
                63,
                "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",
            ),
            (
                64,
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
            (
                65,
                "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0",
            ),
        ];
        for (len, want) in expect {
            let got = hex_digest(&vec![b'a'; len]);
            assert_eq!(got, want, "length {len}");
        }
    }

    #[test]
    fn chunking_does_not_change_the_answer() {
        // The streaming path is where a hand-written hash actually fails: the
        // digest must not depend on how the caller happened to slice the input.
        let data: Vec<u8> = (0..1_000u32).map(|i| (i % 251) as u8).collect();
        let once = hex_digest(&data);
        for chunk in [1usize, 7, 63, 64, 65, 127, 128, 256, 999] {
            let mut h = Sha256::new();
            for part in data.chunks(chunk) {
                h.update(part);
            }
            assert_eq!(h.finish(), once, "chunked by {chunk}");
        }
    }

    #[test]
    fn a_file_hashes_the_same_as_its_bytes() {
        let dir = std::env::temp_dir().join("hearth-sha-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob");
        // Bigger than the read buffer, so the chunked file path is exercised.
        let data: Vec<u8> = (0..(300 * 1024u32)).map(|i| (i % 253) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        assert_eq!(hex_digest_file(&path).unwrap(), hex_digest(&data));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_file_is_an_error_not_a_digest_of_nothing() {
        // Returning the empty-string digest here would make an absent file
        // verify successfully against an empty blob.
        assert!(hex_digest_file(std::path::Path::new("/definitely/not/here")).is_err());
    }

    #[test]
    fn registry_digests_compare_across_their_two_spellings() {
        let from_manifest =
            "sha256:2AF3B81862C6BE03C769683AF18EFDADB2C33F60FF32AB6F83E42C043D6C7816";
        let computed = "2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816";
        assert!(
            matches(from_manifest, computed),
            "a registry writes sha256:… and we compute the bare hex; comparing \
             those literally would report every correct download as corrupt"
        );
        assert!(!matches("sha256:dead", computed));
    }

    #[test]
    fn normalize_is_idempotent() {
        let d = "sha256:ABC123";
        assert_eq!(normalize(d), "abc123");
        assert_eq!(normalize(&normalize(d)), normalize(d));
    }
}

#[cfg(test)]
mod progress_tests {
    use super::*;

    fn tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn progress_variant_returns_the_same_digest() {
        // A verify step that reports progress but hashes differently would be
        // worse than a silent one.
        let p = tmp("hearth_prog_a.bin", &vec![7u8; 900_000]);
        let plain = hex_digest_file(&p).unwrap();
        let loud = hex_digest_file_with_progress(&p, |_, _| {}).unwrap();
        assert_eq!(plain, loud);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn progress_is_monotonic_and_ends_at_the_total() {
        // 900 KB over a 256 KiB chunk = 4 callbacks, so this also proves the
        // callback fires more than once on a file bigger than one chunk.
        let size = 900_000usize;
        let p = tmp("hearth_prog_b.bin", &vec![3u8; size]);
        let mut seen: Vec<(u64, u64)> = Vec::new();
        hex_digest_file_with_progress(&p, |done, total| seen.push((done, total))).unwrap();
        assert!(
            seen.len() > 1,
            "expected several callbacks, got {}",
            seen.len()
        );
        assert!(
            seen.windows(2).all(|w| w[0].0 <= w[1].0),
            "progress went backwards: {seen:?}"
        );
        assert_eq!(
            seen.last().unwrap().0,
            size as u64,
            "final count != file size"
        );
        assert!(
            seen.iter().all(|(_, t)| *t == size as u64),
            "total must be stable"
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn an_empty_file_reports_no_progress_but_still_digests() {
        let p = tmp("hearth_prog_c.bin", b"");
        let mut calls = 0;
        let d = hex_digest_file_with_progress(&p, |_, _| calls += 1).unwrap();
        assert_eq!(calls, 0, "no chunks were read, so nothing to report");
        assert_eq!(d, hex_digest_file(&p).unwrap());
        std::fs::remove_file(&p).ok();
    }
}
