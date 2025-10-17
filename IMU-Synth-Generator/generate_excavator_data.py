#!/usr/bin/env python3
"""
Generate Excavator IMU Data - Simple Script
Tạo dữ liệu IMU cho máy xúc với các tham số đã tune
"""

from imu_synth_generator import IMUSynthGenerator
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Generate Excavator IMU Synthetic Data'
    )
    parser.add_argument(
        '--date',
        type=str,
        default='2025-08-04',
        help='Date to generate data for (YYYY-MM-DD format, default: 2025-08-04)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days (default: 1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV filename (default: auto-generated based on date)'
    )
    parser.add_argument(
        '--morning_start',
        type=str,
        default='07:00',
        help='Morning shift start time (HH:MM format, default: 07:00)'
    )
    parser.add_argument(
        '--morning_end',
        type=str,
        default='09:45',
        help='Morning shift end time (HH:MM format, default: 09:45)'
    )
    parser.add_argument(
        '--afternoon_start',
        type=str,
        default='13:45',
        help='Afternoon shift start time (HH:MM format, default: 13:45)'
    )
    parser.add_argument(
        '--afternoon_end',
        type=str,
        default='17:00',
        help='Afternoon shift end time (HH:MM format, default: 17:00)'
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        date_str = args.date.replace('-', '')
        args.output = f'{date_str}_sensor_data.csv'
    
    # Convert date to UTC start time
    start_utc = f'{args.date} 00:07:00+00:00'
    
    print('🚜 Excavator IMU Data Generator')
    print('=' * 60)
    print(f'Date:       {args.date}')
    print(f'Days:       {args.days}')
    print(f'Seed:       {args.seed}')
    print(f'Output:     {args.output}')
    print(f'Morning:   {args.morning_start} - {args.morning_end}')
    print(f'Afternoon: {args.afternoon_start} - {args.afternoon_end}')
    print()
    
    # Create generator
    gen = IMUSynthGenerator(seed=args.seed)
    
    # Generate data
    print('⏳ Generating data...')
    df = gen.generate(
        start_utc=start_utc,
        days=args.days,
        morning_start=args.morning_start,
        morning_end=args.morning_end,
        afternoon_start=args.afternoon_start,
        afternoon_end=args.afternoon_end
    )
    
    # Save to CSV
    print(f'💾 Saving to {args.output}...')
    df.to_csv(args.output, index=False, na_rep='NaN')
    
    # Show statistics
    active = df[df['Accel_x'].notna()]
    if len(active) > 0:
        print()
        print('✅ Generation Complete!')
        print('=' * 60)
        print(f'Total samples:  {len(df):,}')
        print(f'Active samples: {len(active):,} ({len(active)/len(df)*100:.1f}%)')
        print()
        print('📊 Statistics (working hours):')
        print(f'  acc_norm:  median={active["acc_norm"].median():.2f} m/s²')
        print(f'             range=[{active["acc_norm"].min():.1f}, {active["acc_norm"].max():.1f}]')
        print(f'  gyro_norm: median={active["gyro_norm"].median():.2f} °/s')
        print(f'             max={active["gyro_norm"].max():.0f} °/s')
        print()
        print('⏰ Working Hours (VN time):')
        print(f'  Morning:   {args.morning_start} - {args.morning_end}')
        print(f'  Afternoon: {args.afternoon_start} - {args.afternoon_end}')
        print()
        print(f'📁 File saved: {args.output}')
    else:
        print('⚠️  No active data generated!')


if __name__ == '__main__':
    main()

