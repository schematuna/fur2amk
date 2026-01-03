from __future__ import annotations

import logging
import sys
from typing import Dict, Optional, Tuple

from .model.AMKData import AMKData

from .MMLWriter import MMLWriter
from .MMLUtil import *

class AMKWriter:
    def __init__(self, amk_data: AMKData, module_path: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.txt: str = ''
        self.amk_data = amk_data

        self.add_amk_header()
        self.add_spc_info()
        self.add_sample_info(module_path)
        self.add_ins_info()
        self.add_volume_tempo_info()
        self.add_echo_info()
        self.add_remote_commands()
        self.convert_mml()

    # Sections
    def add_amk_header(self) -> None:
        self.txt += f'#amk {self.amk_data.version}\n\n'

    def add_spc_info(self) -> None:
        # Emit AddmusicK readme-style #spc block with #title/#game/#author/#length
        lines = ['#spc', '{']
        info_align_width = 8
        info = self.amk_data.spc_info
        if info.title:
            lines.append(f'    {'#title':<{info_align_width}} "{info.title}"')
        if info.game:
            lines.append(f'    {'#game':<{info_align_width}} "{info.game}"')
        if info.author:
            lines.append(f'    {'#author':<{info_align_width}} "{info.author}"')
        if info.length:
            lines.append(f'    {'#length':<{info_align_width}} "{info.length}"')
        # Optional comment: use first line of Message if present
        msg = (info.comment or '').strip()
        if msg:
            first_line = msg.splitlines()[0]
            lines.append(f'    {'#comment':<{info_align_width}} "{first_line}"')
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_sample_info(self, path_name: str) -> None:
        sample_lines = [f'#path "{path_name}"', '', '#samples', '{', '    #optimized']
        for _, (samp_name, _) in self.amk_data.samples.items():
            brr_rel = f'{samp_name}'
            sample_lines.append(f'    "{brr_rel}"')
        sample_lines.append('}')
        self.txt += '\n'.join(sample_lines) + '\n\n'

    def add_ins_info(self) -> None:
        if not self.amk_data.instruments:
            return
        lines = ['#instruments', '{']
        # Assign AMK instrument numbers starting at 30 in the order we emit
        # Map of (instrument_index, sample_index) -> AMK instrument number
        self.insnum_map: Dict[Tuple[int, Optional[int]], int] = {}
        next_num = 30
        name_col = max(len(name) for name, _ in self.amk_data.samples.values())
        # get max sample name length for alignment
        name_field_width = name_col + 2  # account for quotes
        # if using sample maps, each sample for an instrument gets its own AMK instrument
        for idx, amk_ins in enumerate(self.amk_data.instruments):
            if amk_ins.is_noise:
                # Noise instrument
                samp_name = f'n{(amk_ins.noise_freq):02X}'
                self.logger.debug(f"Emitting noise instrument {samp_name} for instrument {MMLUtil.to_hex(idx)}.")
            else:
                # Resolve sample filename and tuning
                samp_entry = self.amk_data.samples[amk_ins.sample_index]
                if not samp_entry:
                    # Fallback to first sample
                    samp_entry = next(iter(self.amk_data.samples.values()), ("Sample1.brr", "$01 $00"))
                samp_name, samp_tuning = samp_entry
                samp_name = f'"{samp_name}"'
            # ADSR/GAIN
            # Default: no envelope -> $00 $00
            da = 0x00
            sr = 0x00
            # Default to no GAIN
            ga = 0x00
            if amk_ins.uses_envelope:
                # ADSR on: build ADSR values
                d = int(amk_ins.envelope.decay or 0)
                a = int(amk_ins.envelope.attack or 0)
                ssv = int(amk_ins.envelope.sustain or 0)
                rv = int(amk_ins.envelope.release or 0)
                da = ((d & 0x7) | 0x8) << 4 | (a & 0xF)
                sr = ((ssv & 0x7) << 5) | (rv & 0x1F)
            else:
                if amk_ins.gain_values:
                    # set primary GAIN to first gain value, other will be handled by remote commands
                    ga = amk_ins.gain_values[0]
                elif amk_ins.gain is not None:
                    ga = amk_ins.gain
                else:
                    self.logger.debug(f"Instrument {idx:02X} uses gain mode but has no SNES gain set; defaulting to 0.")
                    ga = 0x00
            lines.append(f'    {samp_name:<{name_field_width}} ${da:02X} ${sr:02X} ${ga:02X} {samp_tuning} ;@{next_num}')
            next_num += 1
        lines.append('}')
        self.txt += '\n'.join(lines) + '\n\n'

    def add_volume_tempo_info(self) -> None:
        self.txt += f'w{self.amk_data.volume} t{self.amk_data.tempo}\n\n'

    def add_echo_info(self) -> None:
        echo_data = self.amk_data.echo_data
        if echo_data:
            self.txt += f'$EF ${MMLUtil.to_hex(echo_data.echoMask)} ${MMLUtil.to_hex(echo_data.echoVolL)} ${MMLUtil.to_hex(echo_data.echoVolR)}\n'
            self.txt += f'$F1 ${MMLUtil.to_hex(echo_data.echoDelay)} ${MMLUtil.to_hex(echo_data.echoFeedback)} ${MMLUtil.to_hex(echo_data.firIdx)}\n'

        if echo_data.echoFilterCoeffs:
            coeffs_hex = ' '.join(f'${MMLUtil.to_hex(c)}' for c in echo_data.echoFilterCoeffs)
            self.txt += f'$F5 {coeffs_hex}\n\n'

    def add_remote_commands(self) -> None:
        for command in self.amk_data.remote_defs:
            self.txt += f"(!{command.command_idx})[{command.amk_command.to_mml()}] ;{command.comment}\n"

        self.txt += "\n\n"

    # Conversion
    def convert_mml(self) -> None:        
        # track global loop labels
        label_count = self.amk_data.label_start
        mml_data = self.amk_data.mml_data
        
        mml_writer = MMLWriter(mml_data, label_count)
        self.txt += mml_writer.write()

        return

    def get_text(self) -> str:
        return self.txt