#!/usr/bin/env python3
"""
Command line interface for 5G NR demodulation.
Allows processing individual files or complete folders.
"""

import sys
import argparse
from pathlib import Path

from nr_demodulator import demodulate_file, demodulate_folder
from config_loader import get_config


def main():
    # Load configuration for defaults
    config = get_config()
    
    parser = argparse.ArgumentParser(
        description='5G NR Demodulator - Processes individual .mat files or complete folders'
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='.mat file or folder with .mat files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='demodulation_results',
        help='Output folder for results (default: demodulation_results)'
    )
    
    parser.add_argument(
        '--scs',
        type=int,
        default=None,
        choices=[15, 30],
        help=f'Subcarrier spacing in kHz (default: {config.scs} from config.yaml)'
    )
    
    parser.add_argument(
        '--gscn',
        type=int,
        default=None,
        help=f'Channel GSCN (default: {config.gscn} from config.yaml)'
    )
    
    parser.add_argument(
        '--lmax',
        type=int,
        default=8,
        help='Number of SSB bursts (default: 8)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mat',
        help='File pattern for folders (default: *.mat)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help="Don't save resource grid images"
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Display detailed processing information (default: silent mode)'
    )
    
    parser.add_argument(
        '--show-axes',
        action='store_true',
        help='Show axes and labels in images (default: no axes)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of threads for parallel processing (default: 4)'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        default='images',
        choices=['images', 'csv', 'both'],
        help='Export format: images (resource grids), csv (demodulated data), or both (default: images)'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"✗ Error: {args.input} does not exist")
        sys.exit(1)
    
    # Determine whether to save images or CSV based on --export
    save_plot = (args.export in ['images', 'both']) and not args.no_plot
    save_csv = (args.export in ['csv', 'both'])
    
    # Process file or folder
    if input_path.is_file():
        result = demodulate_file(
            str(input_path),
            scs=args.scs,
            gscn=args.gscn,
            lmax=args.lmax,
            output_folder=args.output,
            save_plot=save_plot,
            save_csv=save_csv,
            verbose=args.verbose,
            show_axes=args.show_axes
        )
        
        if result:
            print(f"\n✓ Processing completed successfully")
            print(f"✓ Results saved in: {args.output}/")
            sys.exit(0)
        else:
            print(f"\n✗ Processing failed")
            sys.exit(1)
    
    elif input_path.is_dir():
        summary = demodulate_folder(
            str(input_path),
            scs=args.scs,
            gscn=args.gscn,
            lmax=args.lmax,
            output_folder=args.output,
            pattern=args.pattern,
            save_plot=save_plot,
            save_csv=save_csv,
            verbose=args.verbose,
            show_axes=args.show_axes,
            num_threads=args.threads
        )
        
        if summary['successful'] > 0:
            print(f"\n✓ Results saved in: {args.output}/")
            sys.exit(0 if summary['failed'] == 0 else 2)
        else:
            print(f"\n✗ All files failed")
            sys.exit(1)
    
    else:
        print(f"✗ Error: {args.input} is not a valid file or folder")
        sys.exit(1)


if __name__ == '__main__':
    main()
