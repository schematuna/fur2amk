from typing import List
import os

class BRRBlock:
    def __init__(self, data: bytes) -> None:
        if len(data) != 9:
            raise ValueError(f"BRR block length {len(data)} is not 9 bytes")
        self.header: int = data[0]
        self.data: bytes = data[1:9]

# BRR class that stores 9-byte blocks, each with a 1-byte header and 8 bytes of data
class BRRData:
    def __init__(self, data: bytes) -> None:
        self.blocks: List[BRRBlock] = []
        n = len(data)
        if n % 9 != 0:
            raise ValueError(f"BRR data length {n} is not a multiple of 9")
        for i in range(0, n, 9):
            block = BRRBlock(data[i:i+9])
            if len(block.data) != 8:
                raise ValueError(f"BRR block at offset {i} is incomplete")
            self.blocks.append(block)

    def __bytes__(self) -> bytes:
        result = bytearray()
        for block in self.blocks:
            result.append(block.header)
            result.extend(block.data)
        return bytes(result)
    
    def __len__(self) -> int:
        return len(self.blocks)

class BRRSample:
    def __init__(self, name: str, index: int, brr_data: bytes, loop_start: int, loop_end: int) -> None:
        self.name = name
        self.index = index
        self.brr_data = BRRData(brr_data)
        self.loop_start = None
        self.loop_end = None

        num_samps = len(self.brr_data) * 16
        # Furnace will provide out-of-bounds loop points sometimes; clamp them here
        # in particular if the loop end is at the very end of the sample, it will be num_samps instead of num_samps-1
        if loop_start is not None:
            loop_start = max(0, min(loop_start, num_samps - 1))
            self.loop_start = int(loop_start // 16)
            
        if loop_end is not None:
            loop_end = max(0, min(loop_end, num_samps - 1)) 
            self.loop_end = int(loop_end // 16)

# reads in native BRR from Furnace sample format and lints it/fixes it as needed for amk
class BRRConverter:
    def _dump_samples_to_brr(self, out_dir: str, samples: List[BRRSample]) -> None:
        for s in samples:
            # Target BRR path
            # Prefix with index to avoid name collisions and keep ordering stable
            fname_base = (f"{s.index:02d}_" + (f"{s.name}".strip() or f"Sample{s.index}")).replace(' ', '_')
            brr_path = os.path.join(out_dir, fname_base + '.brr')
            # Always overwrite existing BRR: remove it first if present
            try:
                if os.path.exists(brr_path):
                    os.remove(brr_path)
            except OSError:
                pass
            # If the sample already contains raw BRR data, wrap it with AMK 2-byte loop header and write
            if s.brr_data:
                data = s.brr_data
                
                self.validate_and_fix_brr_data(data, s.loop_end)

                loop_off = 0
                if s.loop_start is not None and s.loop_start >= 0:
                    # Convert loop start (samples) to BRR byte offset
                    loop_off = s.loop_start * 9

                header = bytes((loop_off & 0xFF, (loop_off >> 8) & 0xFF))
                with open(brr_path, 'wb') as f:
                    f.write(header + bytes(data))
                print(f"[diag] wrote BRR (raw+hdr): {os.path.basename(brr_path)} loop_off={loop_off}, len={len(data.blocks)+2}")
                continue

            else:
                print(f"[diag] info: sample {s.index:02d} {s.name} has no raw BRR data, skipping")
            
    def validate_and_fix_brr_data(self, data: BRRData, loop_end: int):
        # check that last block has end flag set
        print(f"[diag] info: validating BRR data, num blocks={len(data.blocks)} loop_end={loop_end}")

        is_looped = loop_end is not None and loop_end > 0

        # loop over every 9-byte block and set loop and end flags appropriately
        # end block can be missing for some furnace BRR samples
        # loop flag is often set on every block 
        # per BRR spec, loop flag only has meaning on a block that also has the end flag set, so it's ok if it's set on other blocks
        for i, block in enumerate(data.blocks):
            loop_flag = (block.header & 0x02) != 0
            end_flag = (block.header & 0x01) != 0

            if is_looped and (i == loop_end):
                if not loop_flag:
                    print(f"[diag] warning: BRR loop end block ({i}) missing loop flag; fixing")
                    block.header |= 2
                if not end_flag:
                    print(f"[diag] warning: BRR loop end block ({i}) missing end flag; fixing")
                    block.header |= 1
            
            if not is_looped and (i >= len(data.blocks)) and not end_flag:
                print(f"[diag] warning: BRR last block ({i}) missing end flag; fixing")
                block.header |= 1
                