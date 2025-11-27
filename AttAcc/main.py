import argparse
import csv
import os
from src.system import *
from src.type import *
from src.config import *
from src.ramulator_wrapper import *

RAMULATOR = True


def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 'g_energy (nJ)',
                'g_dram_energy', 'g_l2_energy', 'g_l1_energy', 'g_reg_energy',
                'g_alu_energy', 'g_fc_mem_energy', 'g_fc_comp_energy',
                'g_attn_mem_energy', 'g_attn_comp_energy', 'g_etc_mem_energy',
                'g_etc_comp_energy', 'g_comm_energy'
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()


def run(system,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None):
    print("---Run simple mode Batch {} Lin {} Lout {} pipe {} parall {}---".
          format(batch, lin, lout, pipe, parallel))
    assert system.model_set, "Need to SetModel"
    perfs = []
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    if output_file is not None:
        write_csv(output_file, perfs)


def main():
    parser = argparse.ArgumentParser(
        description="Model configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## set system configuration
    parser.add_argument(
        "--system",
        type=str,
        default="dgx",
        help="dgx (each GPU has 80GB HBM), \
              dgx-cpu (In dgx, offloading the attention layer to cpu), \
              dgx-attacc (dgx + attacc)")
    parser.add_argument(
        "--gpu",
        type=str,
        default='A100a',
        help="GPU type (A100a and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu",
                        type=int,
                        default=8,
                        help="number of GPUs in DGX system. default=8")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB). default=80")



    ## set attacc configuration
    parser.add_argument("--pim",
                        type=str,
                        default='bank',
                        help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--powerlimit",
                        action='store_true',
                        help="power constraint for PIM ")
    parser.add_argument("--ffopt",
                        action='store_true',
                        help="apply feedforward parallel optimization")
    parser.add_argument("--pipeopt",
                        action='store_true',
                        help="apply pipeline optimization ")


    ## set model and service environment
    parser.add_argument(
        "--model",
        type=str,
        default='GPT-175B',
        help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word",
                        type=int,
                        default='2',
                        help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin", "--seqlen",  # Allow both for compatibility
                        type=int,
                        default=2048,
                        dest='lin',
                        help="input sequence length")
    parser.add_argument("--lout",
                        type=int,
                        default=128,
                        help="number of generated tokens")
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help=
        "batch size, default = 1"
    )
    
    # Additional gen_trace_mos.py compatible options
    parser.add_argument("--nhead",
                        type=int,
                        default=32,
                        help="number of attention heads")
    parser.add_argument("--dhead",
                        type=int,
                        default=128,
                        help="head dimension")
    parser.add_argument("--dbyte",
                        type=int,
                        default=2,
                        help="data byte size")
    parser.add_argument(
        "--page_mode",
        type=str,
        default="all",
        choices=['all','even','odd','first_half','second_half','exclude_back_half','ori'],
        help="Page selection mode")
    parser.add_argument(
        "--page_list",
        type=str,
        default="",
        help="Comma-separated list of page indices")
    
    # Mod alignment parameters
    parser.add_argument(
        "--mod_score", 
        type=int, 
        default=1,
        choices=[1, 2, 8],
        help="Mod alignment value for score regions, default=1"
    )
    parser.add_argument(
        "--mod_context",
        type=int,
        default=1, 
        choices=[1, 2, 8],
        help="Mod alignment value for context regions, default=1"
    )
    parser.add_argument(
        "--mac_mode",
        type=str,
        default="buffer",
        choices=['buffer', 'bank', 'bg'], 
        help="MAC operation mode, default=buffer"
    )
    parser.add_argument(
        "--page_scheduling",
        type=str,
        default="false",
        choices=['true', 'false'],
        help="Enable page scheduling"
    )
    parser.add_argument(
        "--page_size",
        type=int,
        default=64,
        help="Page size in tokens, default=64"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="",
        help="CSV file with page selection data"
    )
    parser.add_argument(
        "--output_trace", "--output",  # Allow both for compatibility
        type=str,
        default="",
        dest='output_trace',
        help="Output trace filename (without path)"
    )

    args = parser.parse_args()

    global RAMULATOR
    if RAMULATOR:
        print("The Ramulator {}".format(RAMULATOR))

    if args.gpu == 'H100':
        gpu_device = GPUType.H100
    elif args.gpu == 'A100a':
        gpu_device = GPUType.A100a
    else:
        assert 0

    if args.system == 'dgx-attacc':
        print("{}: ({} x {}), PIM:{}, [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu, args.pim,
            [args.lin, args.lout, args.batch]))
    else:
        print("{}: ({} x {}), [Lin, Lout, batch]: {}".format(
            args.system, args.gpu, args.ngpu,
            [args.lin, args.lout, args.batch]))
    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024
    output_path = "output.csv"
    # Note: Keep existing output.csv for accumulation across different runs
    # if os.path.exists(output_path):
    #     os.system("rm " + output_path)

    # set system
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    system = System(xpu_config['GPU'], modelinfos)
    if args.system in ['dgx-attacc']:
        if args.pim == "bg":
            pim_type = PIMType.BG
        elif args.pim == "buffer":
            pim_type = PIMType.BUFFER
        else:
            pim_type = PIMType.BA
        pim_config = make_pim_config(pim_type,
                                     InterfaceType.NVLINK3,
                                     power_constraint=args.powerlimit)
        
        # Prepare mod alignment parameters
        mod_params = {
            'mod_score': args.mod_score,
            'mod_context': args.mod_context,
            'mac_mode': args.mac_mode,
            'page_scheduling': args.page_scheduling.lower() == 'true',
            'page_size': args.page_size,
            'csv_file': args.csv_file,
            'output_trace': args.output_trace,
            'nhead': args.nhead,
            'dhead': args.dhead,
            'dbyte': args.dbyte,
            'page_mode': args.page_mode,
            'page_list': args.page_list,
            'use_ori': args.page_mode == 'ori'
        }
        
        system.set_accelerator(modelinfos, DeviceType.PIM, pim_config, mod_params)

    elif args.system in ['dgx-cpu']:
        xpu_config = make_xpu_config(gpu_device)
        system.set_xpu(xpu_config['GPU'])
        system.set_accelerator(modelinfos, DeviceType.CPU, xpu_config['CPU'])

    run(system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit)


if __name__ == "__main__":
    main()
