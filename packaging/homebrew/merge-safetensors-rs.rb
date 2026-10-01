# Generated from packaging/homebrew/merge-safetensors-rs.rb in apiplant/merge-safetensors-rs by the
# release workflow, which fills in the version and checksums and commits the
# result to apiplant/homebrew-tap as Formula/merge-safetensors-rs.rb. Changes belong in
# the source repository: the next release overwrites this file.
class MergeSafetensorsRs < Formula
  desc "Merge sharded .safetensors files into one, streaming"
  homepage "https://github.com/apiplant/merge-safetensors-rs"
  version "@VERSION@"
  license "MIT"

  # No bottles: the release archives *are* the binaries, so the formula only
  # unpacks what the tagged workflow already built for each platform.
  on_macos do
    on_arm do
      url "https://github.com/apiplant/merge-safetensors-rs/releases/download/v@VERSION@/merge-safetensors-rs-v@VERSION@-aarch64-apple-darwin.tar.gz"
      sha256 "@SHA_MACOS_ARM64@"
    end
  end
  on_linux do
    on_intel do
      url "https://github.com/apiplant/merge-safetensors-rs/releases/download/v@VERSION@/merge-safetensors-rs-v@VERSION@-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_X86_64@"
    end
    on_arm do
      url "https://github.com/apiplant/merge-safetensors-rs/releases/download/v@VERSION@/merge-safetensors-rs-v@VERSION@-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "@SHA_LINUX_ARM64@"
    end
  end

  def install
    bin.install "merge-safetensors"
    doc.install "README.md"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/merge-safetensors --version")
  end
end
