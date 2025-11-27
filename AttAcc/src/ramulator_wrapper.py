import pandas as pd
import subprocess
import math
import os
from src.config import *
from src.model import *
from src.type import *


class Ramulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=5,
                 mod_score=1,
                 mod_context=1,
                 mac_mode='buffer',
                 page_scheduling=False,
                 page_size=64,
                 csv_file='',
                 output_trace='',
                 use_ori=False):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
        self.fast_mode = fast_mode
        
        # Mod alignment parameters
        self.mod_score = mod_score
        self.mod_context = mod_context
        self.mac_mode = mac_mode
        self.page_scheduling = page_scheduling
        self.page_size = page_size
        self.csv_file = csv_file
        self.output_trace = output_trace
        self.use_ori = use_ori
        
        self.current_csv_filename = "none"  # Store current CSV filename for logging
        self.initial_seqlen = None  # Store initial sequence length for trace generation
        self.trace_generated = False  # Flag to prevent multiple trace generation
        self.generated_traces = {}  # Cache of generated trace results

    def make_yaml_file(self, yaml_file, file_name, power_constraint):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")
        line = ""
        line += "Frontend:\n"
        line += "  impl: PIMLoadStoreTrace\n"
        line += "  path: {}\n".format(trace_path)
        line += "  clock_ratio: 1\n"
        line += "\n"
        line += "  Translation:\n"
        line += "    impl: NoTranslation\n"
        line += "    max_addr: 2147483648\n"
        line += "              \n"
        line += "\n"
        line += "MemorySystem:\n"
        line += "  impl: PIMDRAM\n"
        line += "  clock_ratio: 1\n"
        line += "  DRAM:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    org:\n"
        line += "      preset: HBM3_8Gb_2R\n"
        line += "      channel: 16\n"
        line += "    timing:\n"
        if power_constraint:
            line += "      preset: HBM3_5.2Gbps\n"
        else:
            line += "      preset: HBM3_5.2Gbps_NPC\n"
        line += "\n"
        line += "  Controller:\n"
        line += "    impl: HBM3-PIM\n"
        line += "    Scheduler:\n"
        line += "      impl: PIM\n"
        line += "    RefreshManager:\n"
        line += "      impl: AllBankHBM3\n"
        line += "      #impl: No\n"
        line += "    plugins:\n"
        line += "\n"
        line += "  AddrMapper:\n"
        line += "    impl: HBM3-PIM\n"
        with open(yaml_file, 'w') as f:
            f.write(line)

    def update_log_file(self, log):
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                columns = [
                    'L', 'nhead', 'dhead', 'dbyte', 'pim_type',
                    'power_constraint', 'page_scheduling', 'mod_score', 'mod_context', 'mac_mode',
                    'cycle', 'mac', 'softmax', 'mvgb', 'mvsb', 'wrgb', 'csv_file'
                ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df
        
        # Ensure all columns exist
        if 'csv_file' not in df.columns:
            df['csv_file'] = 'none'
        if 'page_scheduling' not in df.columns:
            df['page_scheduling'] = False
        if 'mod_score' not in df.columns:
            df['mod_score'] = 2
        if 'mod_context' not in df.columns:
            df['mod_context'] = 8
        if 'mac_mode' not in df.columns:
            df['mac_mode'] = 'buffer'
        
        # Add all configuration info to log
        # Original log format: [l, nhead, dhead, dbyte, pim_type, power_constraint, cycle, mac, softmax, mvgb, mvsb, wrgb]
        # New format: [l, nhead, dhead, dbyte, pim_type, power_constraint, page_scheduling, mod_score, mod_context, mac_mode, cycle, mac, softmax, mvgb, mvsb, wrgb, csv_file]
        log_extended = (log[:6] + 
                       [self.page_scheduling, self.mod_score, self.mod_context, self.mac_mode] + 
                       log[6:] + 
                       [self.current_csv_filename])
        
        # Check if same condition already exists and remove it (to overwrite)
        l, nhead, dhead, dbyte, pim_type, power_constraint = log[:6]
        mask = (df['nhead'] == nhead) & (df['dhead'] == dhead) & \
               (df['dbyte'] == dbyte) & (df['pim_type'] == pim_type) & \
               (df['power_constraint'] == power_constraint) & \
               (df['page_scheduling'] == self.page_scheduling) & \
               (df['mod_score'] == self.mod_score) & \
               (df['mod_context'] == self.mod_context) & \
               (df['mac_mode'] == self.mac_mode) & \
               (df['csv_file'] == self.current_csv_filename)
        
        df = df[~mask]  # Remove existing rows with same condition
        
        # Add new row
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log_extended
        df = pd.concat([df, new_df], ignore_index=True)
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    #def run_ramulator(self):
    def run_ramulator(self, pim_type: PIMType, l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name):
        pim_type_name = pim_type.name.lower(
        ) if not pim_type == PIMType.BA else "bank"
        
        # Create a unique key based on model configuration (not sequence length)
        config_key = f"nattn{self.nhead}_dhead{self.dhead}_mod{self.mod_score}_{self.mod_context}_{self.mac_mode}_{pim_type_name}_ps{self.page_scheduling}"
        
        # Check if we already have result for this configuration
        if config_key in self.generated_traces:
            print(f"🔄 Using cached trace result for: {config_key}")
            return self.generated_traces[config_key]
        
        # Always use unique trace filename based on parameters
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        # Use different trace generation script based on ori flag
        if self.use_ori:
            trace_exc = os.path.join(
                self.ramulator_dir,
                "trace_gen/gen_trace_ori.py")
            print("🔧 Using original bank-level AttAcc trace generation (gen_trace_ori.py)")
        else:
            trace_exc = os.path.join(
                self.ramulator_dir,
                "trace_gen/gen_trace_mos.py")
            print("🔧 Using MoS-based AttAcc trace generation (gen_trace_mos.py)")
        
        # Use provided CSV file or default
        if self.csv_file:
            if os.path.isabs(self.csv_file):
                csv_file = self.csv_file
            else:
                csv_file = os.path.join(self.ramulator_dir, "trace_gen", self.csv_file)
        else:
            csv_file = os.path.join(
                self.ramulator_dir,
                "trace_gen/2_page_selection_log_20251013_194057_selected_pages.csv")
        
        # Extract CSV filename for logging
        self.current_csv_filename = os.path.basename(csv_file) if csv_file else "none"
        
        # Build trace generation command with mod alignment parameters
        # Store initial sequence length on first call, then use it for all subsequent calls
        if self.initial_seqlen is None:
            self.initial_seqlen = l
        
        # Build trace arguments based on ori flag
        if self.use_ori:
            # Use gen_trace_ori.py argument format (based on gen_trace_attacc_bank.py)
            trace_args = "--seqlen {} --nhead {} --dhead {} --dbyte {} --output {} --page_size {}".format(
                self.initial_seqlen, self.nhead, self.dhead, dbyte, trace_file, self.page_size)
        else:
            # Use gen_trace_mos.py argument format
            trace_args = "--seqlen {} --nhead {} --dhead {} --dbyte {} --output {} --page_size {}".format(
                self.initial_seqlen, self.nhead, self.dhead, dbyte, trace_file, self.page_size)
        
        # Determine maxlen based on CSV file or use reasonable default
        maxlen_value = 4  # default
        if csv_file and os.path.exists(csv_file):
            trace_args += " --csv_file {}".format(csv_file)
            # Try to determine time steps from CSV if possible
            try:
                import pandas as pd
                df = pd.read_csv(csv_file)
                if 'time_step' in df.columns:
                    maxlen_value = df['time_step'].max() + 1
                    print(f"📊 CSV file has {maxlen_value} time steps, setting maxlen={maxlen_value}")
                else:
                    maxlen_value = 4  # reasonable default for CSV case
                    print(f"📊 CSV file provided but no time_step column, using maxlen={maxlen_value}")
            except Exception as e:
                print(f"⚠️ Error reading CSV file: {e}, using default maxlen={maxlen_value}")
        
        if not self.use_ori:
            # Only add maxlen and mod alignment parameters for gen_trace_mos.py
            trace_args += " --maxlen {}".format(maxlen_value)
            trace_args += " --mod_score {} --mod_context {}".format(self.mod_score, self.mod_context)
            trace_args += " --mac_mode {}".format(self.mac_mode)
            trace_args += " --page_scheduling {}".format('true' if self.page_scheduling else 'false')
        else:
            # For gen_trace_ori.py, add maxlen (required) but keep other params minimal
            trace_args += " --maxlen {}".format(max(maxlen_value, 4096))  # Use reasonable maxlen for ori

        gen_trace_cmd = f"python {trace_exc} {trace_args}"
        
        print(f"🔧 Generating trace with command: {gen_trace_cmd}")
        
        # generate trace with better error handling
        try:
            result = subprocess.run(gen_trace_cmd, 
                                  shell=True, 
                                  capture_output=True, 
                                  text=True)
            
            if result.returncode != 0:
                print(f"❌ Trace generation failed!")
                print(f"Command: {gen_trace_cmd}")
                print(f"Return code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise Exception(f"Trace generation failed with return code {result.returncode}")
            else:
                print(f"✅ Trace generated successfully: {trace_file}")
                if not os.path.exists(trace_file):
                    print(f"⚠️  Warning: Trace file {trace_file} was not created!")
                    raise Exception(f"Trace file {trace_file} does not exist after generation")
                    
        except Exception as e:
            print(f"Error generating trace: {e}")
            raise

        # run ramulator
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    shell=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # remove trace
        rm_trace_cmd = f"rm {trace_file}"
        try:
            os.system(rm_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # parsing output
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        out = [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
            n_cmds["wrgb"]
        ]
        
        # Cache the result to prevent repeated trace generation
        self.generated_traces[config_key] = out
        print(f"💾 Cached result for configuration: {config_key}")
        self.generated_traces[config_key] = out
        print(f"✅ Cached trace result for: {config_key}")
        
        return out

    def run(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            # Store initial sequence length on first call
            if self.initial_seqlen is None:
                self.initial_seqlen = l
                print(f"🔧 Setting initial sequence length: {self.initial_seqlen}")

            # Add page_scheduling suffix and mod info to file name
            # Use initial sequence length for consistent naming
            ps_suffix = "s" if self.page_scheduling else ""
            file_name = "attacc_l{}_nattn{}_dhead{}_dbyte{}_pc{}{}_mod{}_{}_{}".format(
                self.initial_seqlen, self.nhead, dhead, layer.dbyte, int(power_constraint), ps_suffix,
                self.mod_score, self.mod_context, self.mac_mode)
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(pim_type, self.initial_seqlen, self.nhead,
                                        layer.dbyte, yaml_file, file_name)

            # remove trace
            rm_yaml_cmd = f"rm {yaml_file}"
            try:
                os.system(rm_yaml_cmd)
            except Exception as e:
                print(f"Error: {e}")

            # post processing
            # 32: read granularity
            cycle, mac, sfm, mvgb, mvsb, wrgb = result
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 1

            ## update log file

            log = [
                l, num_ops_per_hbm, dhead, dbyte, pim_type.name,
                power_constraint
            ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if self.df.empty:
            self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = layer.numOp
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        l = layer.n
        dhead = layer.k
        dbyte = layer.dbyte
        # Include CSV filename and mod parameters in the condition check
        # Use initial sequence length and model parameters for caching
        search_seqlen = self.initial_seqlen if self.initial_seqlen is not None else l
        
        if 'csv_file' in self.df.columns and 'page_scheduling' in self.df.columns:
            # New format with all parameters
            row = self.df[(self.df['L'] == search_seqlen) & \
                          (self.df['nhead'] == self.nhead) & \
                          (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                          (self.df['power_constraint'] == power_constraint) &  \
                          (self.df['pim_type'] == pim_type.name) & \
                          (self.df['page_scheduling'] == self.page_scheduling) & \
                          (self.df['mod_score'] == self.mod_score) & \
                          (self.df['mod_context'] == self.mod_context) & \
                          (self.df['mac_mode'] == self.mac_mode) & \
                          (self.df['csv_file'] == self.current_csv_filename)]
        else:
            # For backward compatibility with old log files
            row = self.df[(self.df['L'] == search_seqlen) & \
                          (self.df['nhead'] == self.nhead) & \
                          (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                          (self.df['power_constraint'] == power_constraint) &  \
                          (self.df['pim_type'] == pim_type.name)]
        if row.empty:
            return self.run(pim_type, layer, power_constraint)

        else:
            cycle = int(row.iloc[0]['cycle'])
            mac = int(row.iloc[0]['mac'])
            softmax = int(row.iloc[0]['softmax'])
            mvgb = int(row.iloc[0]['mvgb'])
            mvsb = int(row.iloc[0]['mvsb'])
            wrgb = int(row.iloc[0]['wrgb'])
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = mac * 32
            if pim_type == PIMType.BA:
                # pCH * Rank * bank group * bank
                mem_acc *= 2 * 2 * 4 * 4
            elif pim_type == PIMType.BG:
                # pCH * Rank * bank group
                mem_acc *= 2 * 2 * 4
            else:
                mem_acc *= 2

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic
