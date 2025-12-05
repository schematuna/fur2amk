# Converts a furnace object to an AMK object

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum, auto

from FurnaceData import FurnaceModule, FurnaceRow
from AMKData import AMKData, AMKInstrument, AMKRemoteCommand, AMKRemoteCommandType, AMKRemoteCommandTiming
from AMKData import Event, EventTable, EventType

# enum for state types
class EventStateType(Enum):
    INST = auto()
    VOLUME = auto()
    PAN = auto()
    EFFECT = auto()
    
# persitent channel state, useful for avoiding repeat emission
class EventState:
    def __init__(self):
        self.state_d = { EventStateType.INST: None,
                         EventStateType.VOLUME: None,
                         EventStateType.PAN: None,
                         EventStateType.EFFECT: {} }


class FurnaceConverter:
    def convert_samples(self, module: FurnaceModule) -> Dict[int, Tuple[str, str]]:
        sample_dict = {}
        for s in module.Samples:
            # Build sample filename and tuning string
            fname = f"{s.index:02d}_" + (s.name or f"Sample{s.index}").replace(' ', '_') + '.brr'
            tuning_word = 0x0100
            if s.c4_rate and s.c4_rate > 0:
                # MAGIC NUMBERS to convert from c4_rate to AMK instrument tuning value
                # stolen from it2amk's SampConv
                val = int(round(float(s.c4_rate) * 768 / 12539))
                tuning_word = max(0, min(0xFFFF, val))
            tune_str = f"${(tuning_word >> 8) & 0xFF:02X} ${(tuning_word & 0xFF):02X}"
            sample_dict[s.index] = (fname, tune_str)

        return sample_dict
    
    def convert_instruments(self, module: FurnaceModule) -> List[AMKInstrument]:
        ins_list = []
        # For each instrument, gather unique samples referenced by its sample map
        for ins in module.Instruments:
            instruments: List[AMKInstrument] = []
            # first, check if this is a noise instrument
            if ins.snes_macro_data.is_noise:
                if ins.snes_macro_data.noise_freq is None:
                    ins.snes_macro_data.noise_freq = 29  # default noise freq if unset
                    print(f"Warning: Instrument {ins.index} is a noise instrument but has no noise frequency set; You should set it explicitly in Furnace.", file=sys.stderr)
                instruments.append(AMKInstrument.noise(index=ins.index, noise_freq=ins.snes_macro_data.noise_freq))
            elif ins.use_sample_map and ins.sample_table:
                # Collect unique, non-zero sample indices from the 120-entry map
                uniq: List[int] = []
                seen = set()
                for (_note_to_play, samp_to_play) in ins.sample_table:
                    # Furnace SM uses 0-based sample indices
                    sidx1 = int(samp_to_play)
                    if sidx1 < 0:
                        continue
                    if sidx1 not in seen:
                        seen.add(sidx1)
                        uniq.append(sidx1)
                for sidx in uniq:
                    instruments.append(AMKInstrument.sample(index=ins.index, sample_index=sidx))
            else:
                # Use initial_sample (0-based) if available
                if ins.initial_sample is not None and int(ins.initial_sample) >= 0:
                    sidx1 = int(ins.initial_sample)
                else:
                    sidx1 = 0
                instruments.append(AMKInstrument.sample(index=ins.index, sample_index=sidx1))
            # Track used samples and populate instrument entries
            for inst in instruments:
                ins_list.append((ins.index, inst))

        return ins_list
    
    def convert_remote_commands(self, module: FurnaceModule, amk_data: AMKData) -> int:
        # start at 1, 0 will be reserved for stop remote commands
        command_num = 1

        # gather remote commands and associate with an instrument
        for ins in module.Instruments:
            gmacro = ins.snes_macro_data.gain_values
            if gmacro and len(gmacro) > 1:
                # just support one gain change for now.
                # I think amk would allow no more than 2 remote commands at once anyways
                # find this instrument in ins_list
                for (ins_index, inst) in amk_data.instruments:
                    if ins_index == ins.index:
                        cmd = AMKRemoteCommand(command_num, 
                                                AMKRemoteCommandTiming.AFTER_START, 
                                                AMKRemoteCommandType.GAIN, 
                                                f"={ins.snes_macro_data.gain_speed}",
                                                [gmacro[1]])
                        inst.remote_commands.append(cmd)
                        command_num += 1

        # need to indicate where to pick up with labels
        # loop labels and remote command labels can't overlap
        return command_num

    def convert_loop_marker(self, module: FurnaceModule) -> int:
        # iterate all rows for command 0Bxx (jump to order)
        # This will be interpreted as the intro marker position
        intro_order = None
        for c in range(module.NumChannels):
            patmap = module.PatternsByChannel[c] if c < len(module.PatternsByChannel) else {}
            orders = module.OrdersPerChannel[c] if c < len(module.OrdersPerChannel) else []
            for pat_idx in orders:
                rows = patmap.get(pat_idx)
                if rows:
                    for row in rows:
                        for effect in (row.Effects or []):
                            if effect[0] == 0x0B:
                                intro_order = int(effect[1])

        return intro_order
    
    def convert_events(self, module: FurnaceModule) -> EventTable:
        event_table = EventTable()

        for ch in range(module.NumChannels):
            flat_rows: List[FurnaceRow] = []
            patmap = module.PatternsByChannel[ch] if ch < len(module.PatternsByChannel) else {}
            orders = module.OrdersPerChannel[ch] if ch < len(module.OrdersPerChannel) else []
            for pat in orders:
                rows = patmap.get(pat)
                if rows:
                    flat_rows.extend(rows)
                else:
                    print(f"Warning: Channel {ch} references missing pattern {pat}. Inserting empty pattern.", file=sys.stderr)
                    flat_rows.extend([FurnaceRow() for _ in range(module.PatternLength)])

            state = self.states[ch].state_d
            tick = 0
            ticksPerRow = module.Speed1
            for row in flat_rows:
                # process row into events
                # Note
                type = row._kind()
                if type == FurnaceRow.NoteKind.NOTE:
                    event_table.events[ch].append(Event(tick, EventType.NOTE, row.Note))
                elif type == FurnaceRow.NoteKind.OFF:
                    event_table.events[ch].append(Event(tick, EventType.NOTE_OFF, None))
                # Instrument
                ins = row.Ins
                if ins is not None:
                    if ins != state[EventStateType.INST]:
                        state[EventStateType.INST] = ins
                        event_table.events[ch].append(Event(tick, EventType.INS_CHANGE, ins))
                # volume
                vol = row.Vol
                if vol is not None:
                    if vol != state[EventStateType.VOLUME]:
                        state[EventStateType.VOLUME] = vol
                        event_table.events[ch].append(Event(tick, EventType.VOLUME, vol))
                # Effects
                # for effect in (row.Effects or []):
                #     eff_num = effect[0]
                #     eff_val = effect[1]
                #     event_table.events[ch].append(Event(tick, f'effect_{eff_num:02X}', eff_val))
                tick += ticksPerRow

    def convert(self, module: FurnaceModule) -> AMKData:
        amk_data = AMKData()

        amk_data.samples     = self.convert_samples(module)
        amk_data.instruments = self.convert_instruments(module)
        amk_data.label_start = self.convert_remote_commands(module, amk_data)
        amk_data.intro_order = self.convert_loop_marker(module)
        amk_data.event_table = self.convert_events(module)

        return amk_data