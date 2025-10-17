#!/usr/bin/env python3
"""
IMU Synth Generator - Tạo dữ liệu IMU tổng hợp theo đặc tả
Generates synthetic IMU data (accelerometer + gyroscope) mimicking real human activity patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import sys
import argparse
from typing import Optional, Tuple


class IMUSynthGenerator:
    """Generator for synthetic IMU time-series data"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize with optional random seed"""
        self.rng = np.random.RandomState(seed)
        
    def generate(
        self,
        start_utc: str,
        days: int = 1,
        mean_dt: float = 0.30,
        jitter: float = 0.02,
        calibrate_from_head5_path: Optional[str] = None,
        morning_start: str = "07:00",
        morning_end: str = "09:45",
        afternoon_start: str = "13:45",
        afternoon_end: str = "17:00"
    ) -> pd.DataFrame:
        """
        Generate synthetic IMU data
        
        Args:
            start_utc: Start time in UTC (e.g., "2025-08-04 00:07:00+00:00")
            days: Number of days to generate
            mean_dt: Mean sampling interval in seconds (default 0.30)
            jitter: Jitter amplitude for sampling interval (default 0.02)
            calibrate_from_head5_path: Optional path to 5-row CSV for calibration
            morning_start: Morning shift start time (HH:MM format, default "07:00")
            morning_end: Morning shift end time (HH:MM format, default "09:45")
            afternoon_start: Afternoon shift start time (HH:MM format, default "13:45")
            afternoon_end: Afternoon shift end time (HH:MM format, default "17:00")
            
        Returns:
            DataFrame with IMU data
        """
        # Calibration from head5 if provided
        if calibrate_from_head5_path:
            mean_dt, R0 = self._calibrate_from_head5(calibrate_from_head5_path)
        else:
            R0 = np.eye(3)
            
        # 1) Build timeline (1 sample per second, no jitter)
        t_utc, t_local, active_mask = self._build_timeline(
            start_utc, days, mean_dt, jitter,
            morning_start, morning_end, afternoon_start, afternoon_end
        )
        N = len(t_utc)
        
        # Fixed sampling rate: 1 sample per second
        dt = 1.0
        
        # 2) Orientation drift (slow rotation)
        R_series = self._generate_orientation_drift(N, dt, R0)
        
        # 3) State machine for activity
        states = self._generate_activity_states(active_mask)
        
        # 4) Synthesize sensor signals
        acc, gyr = self._synthesize_sensors(
            N, R_series, states, active_mask, dt
        )
        
        # 5) Add noise, bias, outliers and clip
        acc, gyr = self._add_noise_and_artifacts(acc, gyr, active_mask, dt)
        
        # 6) Compute derived columns
        acc_norm = np.sqrt(np.sum(acc**2, axis=1))
        gyro_norm = np.sqrt(np.sum(gyr**2, axis=1))
        
        # 7) Build dataframe
        df = self._build_dataframe(
            t_utc, t_local, acc, gyr, acc_norm, gyro_norm, N
        )
        
        return df
    
    def _calibrate_from_head5(self, path: str) -> Tuple[float, np.ndarray]:
        """Calibrate mean_dt and initial rotation from 5-row sample"""
        df = pd.read_csv(path, nrows=5)
        
        # Estimate mean_dt from timestamps
        timestamps = pd.to_datetime(df['Timestamp'])
        diffs = timestamps.diff().dt.total_seconds().dropna()
        mean_dt = diffs.median()
        
        # Estimate initial rotation from mean acceleration
        acc_cols = ['Accel_x', 'Accel_y', 'Accel_z']
        mean_acc = df[acc_cols].mean().values
        
        # Align measured gravity with world gravity [0, 0, -9.81] (Z-axis down)
        g_world = np.array([0, 0, -9.81])
        mean_acc_norm = mean_acc / np.linalg.norm(mean_acc)
        g_world_norm = g_world / np.linalg.norm(g_world)
        
        # Rotation to align
        v = np.cross(g_world_norm, mean_acc_norm)
        c = np.dot(g_world_norm, mean_acc_norm)
        
        if np.abs(c + 1.0) < 1e-6:  # 180 degree rotation
            R0 = -np.eye(3)
        elif np.abs(c - 1.0) < 1e-6:  # No rotation needed
            R0 = np.eye(3)
        else:
            # Rodrigues' rotation formula
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R0 = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
            
        return mean_dt, R0
    
    def _build_timeline(
        self, 
        start_utc: str, 
        days: int, 
        mean_dt: float, 
        jitter: float,
        morning_start: str = "07:00",
        morning_end: str = "09:45",
        afternoon_start: str = "13:45",
        afternoon_end: str = "17:00"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build timeline ONLY for working hours (machine only logs when operating)"""
        # Parse start time and convert to VN timezone
        start_date = pd.Timestamp(start_utc).tz_convert('UTC')
        tz_local = pytz.timezone('Asia/Ho_Chi_Minh')
        
        # Parse working hours
        morning_start_hour, morning_start_min = map(int, morning_start.split(':'))
        morning_end_hour, morning_end_min = map(int, morning_end.split(':'))
        afternoon_start_hour, afternoon_start_min = map(int, afternoon_start.split(':'))
        afternoon_end_hour, afternoon_end_min = map(int, afternoon_end.split(':'))
        
        # Generate timestamps only for working hours
        t_utc_list = []
        t_local_list = []
        
        for day in range(days):
            day_offset = timedelta(days=day)
            base_date = start_date + day_offset
            
            # Morning shift: customizable start - end VN time
            morning_start_local = tz_local.localize(
                datetime(base_date.year, base_date.month, base_date.day, 
                       morning_start_hour, morning_start_min, 0)
            )
            morning_end_local = tz_local.localize(
                datetime(base_date.year, base_date.month, base_date.day, 
                       morning_end_hour, morning_end_min, 0)
            )
            morning_seconds = int((morning_end_local - morning_start_local).total_seconds())
            
            for i in range(morning_seconds):
                t_local = morning_start_local + timedelta(seconds=i)
                t_utc_list.append(t_local.astimezone(pytz.UTC))
                t_local_list.append(t_local)
            
            # Afternoon shift: customizable start - end VN time
            afternoon_start_local = tz_local.localize(
                datetime(base_date.year, base_date.month, base_date.day, 
                       afternoon_start_hour, afternoon_start_min, 0)
            )
            afternoon_end_local = tz_local.localize(
                datetime(base_date.year, base_date.month, base_date.day, 
                       afternoon_end_hour, afternoon_end_min, 0)
            )
            afternoon_seconds = int((afternoon_end_local - afternoon_start_local).total_seconds())
            
            for i in range(afternoon_seconds):
                t_local = afternoon_start_local + timedelta(seconds=i)
                t_utc_list.append(t_local.astimezone(pytz.UTC))
                t_local_list.append(t_local)
        
        t_utc = np.array(t_utc_list)
        t_local = np.array(t_local_list)
        
        # All samples are active (no mask needed, but return True array for compatibility)
        active_mask = np.ones(len(t_utc), dtype=bool)
        
        return t_utc, t_local, active_mask
    
    def _compute_working_hours_mask(self, t_local: np.ndarray) -> np.ndarray:
        """Determine which samples are in working hours (excavator schedule)"""
        active = np.zeros(len(t_local), dtype=bool)
        
        for i, t in enumerate(t_local):
            hour = t.hour
            minute = t.minute
            time_minutes = hour * 60 + minute
            
            # Excavator working hours: 07:06-09:45 and 13:45-17:01 (include 17:01, exclude after)
            morning = (7*60 + 6 <= time_minutes < 9*60 + 45)
            afternoon = (13*60 + 45 <= time_minutes <= 17*60 + 1)  # Until 17:01 inclusive
            
            active[i] = morning or afternoon
            
        return active
    
    def _generate_orientation_drift(
        self, 
        N: int, 
        dt: float, 
        R0: np.ndarray
    ) -> np.ndarray:
        """Generate slow orientation drift over time"""
        # Base angular rates (very minimal drift to keep acc_norm near 9.81)
        # We want angles to stay within ~5-10 degrees maximum
        sigma_rate = 0.0003  # rad/s (reduced 10x)
        base_rates = self.rng.normal(0, sigma_rate, size=(N, 3))
        
        # Smooth with moving average to create slow drift (20-60 min periods)
        window = int(20 * 60 / dt)  # ~20 minutes
        for axis in range(3):
            base_rates[:, axis] = np.convolve(
                base_rates[:, axis], 
                np.ones(window)/window, 
                mode='same'
            )
        
        # Integrate to get angles, but limit to small angles
        angles = np.cumsum(base_rates * dt, axis=0)
        # Limit angles to ±0.10 rad (~±5.7 degrees) to keep acc_norm in 6-16 range
        angles = np.clip(angles, -0.10, 0.10)
        
        # Create rotation matrices
        R_series = np.zeros((N, 3, 3))
        for i in range(N):
            yaw, pitch, roll = angles[i]
            R_series[i] = self._euler_to_rotation(yaw, pitch, roll) @ R0
            
        return R_series
    
    def _euler_to_rotation(self, yaw: float, pitch: float, roll: float) -> np.ndarray:
        """Convert Euler angles (ZYX convention) to rotation matrix"""
        # Rz(yaw) * Ry(pitch) * Rx(roll)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        
        return Rz @ Ry @ Rx
    
    def _generate_activity_states(
        self, 
        active_mask: np.ndarray
    ) -> np.ndarray:
        """Generate activity state sequence"""
        N = len(active_mask)
        states = np.zeros(N, dtype=int)  # 0=idle, 1=walk, 2=turn, 3=burst
        
        # State probabilities - excavator for target distribution
        state_probs = {'idle': 0.32, 'digging': 0.55, 'rotating': 0.10, 'moving': 0.03}
        state_map = {'idle': 0, 'digging': 1, 'rotating': 2, 'moving': 3}
        
        # Duration range: varies by state (dt = 1 second per sample)
        # Real data shows: idle = long periods, bursts = very short
        dt = 1.0  # 1 sample per second
        min_dur_samples = int(30 / dt)
        max_dur_samples = int(8 * 60 / dt)
        
        i = 0
        while i < N:
            if not active_mask[i]:
                states[i] = 0  # idle when not active
                i += 1
                continue
                
            # Choose state
            state_name = self.rng.choice(
                list(state_probs.keys()), 
                p=list(state_probs.values())
            )
            state_id = state_map[state_name]
            
            # Choose duration based on excavator activity
            if state_name == 'moving':
                # Moving: short, 5-15 seconds
                duration = self.rng.randint(int(5/dt), int(15/dt))
            elif state_name == 'rotating':
                # Rotating boom: 10-25 seconds (longer for more plateau samples)
                duration = self.rng.randint(int(10/dt), int(25/dt))
            elif state_name == 'digging':
                # Digging operation: 20-50 seconds (longer for plateau)
                duration = self.rng.randint(int(20/dt), int(50/dt))
            else:
                # Idle: 20s-3min (longer for more 0°/s samples)
                duration = self.rng.randint(int(20/dt), int(3*60/dt))
            
            # Fill state, respecting active_mask
            for j in range(duration):
                if i + j >= N:
                    break
                if active_mask[i + j]:
                    states[i + j] = state_id
                    
            i += duration
            
        return states
    
    def _synthesize_sensors(
        self,
        N: int,
        R_series: np.ndarray,
        states: np.ndarray,
        active_mask: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synthesize accelerometer and gyroscope signals"""
        acc = np.zeros((N, 3))
        gyr = np.zeros((N, 3))
        
        g_world = np.array([0, 0, -9.81])  # Z-axis down (gravity = -9.81)
        
        # Generate walking oscillations (pre-compute for efficiency)
        walk_freq = 1.9  # Hz (1.6-2.2 range)
        time_vec = np.arange(N) * dt
        walk_osc = np.sin(2 * np.pi * walk_freq * time_vec)
        
        for i in range(N):
            if not active_mask[i]:
                # Outside working hours: set to NaN
                acc[i, :] = np.nan
                gyr[i, :] = np.nan
                continue
                
            # Gravity component in body frame
            g_body = R_series[i] @ g_world
            
            state = states[i]
            
            if state == 0:  # idle - excavator engine running, stationary
                # Accel_x: mean=2, range [-10, 12], very low variation for natural bell curve
                acc_add = self.rng.normal(2, 0.8, 3)  # Mean=2, very low variance for natural bell curve
                # Accel_y: mean=0, range [-5, 5], low variation
                acc_add[1] = self.rng.normal(0, 0.5, 1)  # Mean=0, range [-5, 5]
                # Accel_z: mean=-9, range [-15, 0], medium variation
                acc_add[2] = self.rng.normal(0, 0.8, 1)  # Z variation around 0, will be added to gravity

                # Gyro: 0°/s peak ~11k, 1°/s ~700-900, 2°/s ~500-650, smooth transition
                gyro_choice = self.rng.uniform()
                if gyro_choice < 0.03:
                    gyr_add = self.rng.normal(0, 0.6, 3)  # ~1°/s (target ~700-900)
                elif gyro_choice < 0.08:
                    gyr_add = self.rng.normal(0, 1.2, 3)  # ~2°/s (target ~500-650)
                elif gyro_choice < 0.15:
                    # Bridge region: 2-6°/s (giảm từ 18% xuống 15% để tăng 0°/s)
                    gyr_add = self.rng.normal(0, 2.2, 3)  # ~3-5°/s
                else:
                    gyr_add = self.rng.normal(0, 0.02, 3)  # ~0°/s (peak ~11k - tăng từ 82% lên 85%)
                
            elif state == 1:  # digging - bucket digging motion
                # Accel_x: mean=2, low variation, range [-10, 12] for natural bell curve
                acc_add = self.rng.normal(2, 1.2, 3)  # Mean=2, low variance for natural bell curve
                # Accel_y: mean=0, range [-5, 5], low variation
                acc_add[1] = self.rng.normal(0, 0.5, 1)  # Mean=0, range [-5, 5]
                # Accel_z: mean=-9, range [-15, 0], medium variation
                acc_add[2] = self.rng.normal(0, 1.2, 1)  # Z variation around 0

                # Gyro: 3-27 plateau ~400/bin, smooth distribution with bridge
                gyro_type = self.rng.uniform(0, 1)
                if gyro_type < 0.08:
                    # 8% in bridge range 2-6°/s (tăng từ 6% lên 8% để tránh lõm)
                    gyro_base = self.rng.uniform(2, 6)
                elif gyro_type < 0.80:
                    # 72% in plateau range 6-27°/s (giảm từ 76% xuống 72%)
                    gyro_base = self.rng.uniform(6, 27)
                elif gyro_type < 0.95:
                    # 15% in 27-40°/s (giữ nguyên)
                    gyro_base = self.rng.uniform(27, 40)
                else:
                    # 2% in 40-65°/s (giảm từ 3% xuống 2% để giảm 40-60°/s)
                    gyro_base = self.rng.uniform(40, 65)
                gyr_add = self.rng.normal(0, 1.0, 3)  # Reduced variance for smoother plateau
                gyr_add[self.rng.choice([0,1,2])] += gyro_base
                    
            elif state == 2:  # rotating - boom/cab rotation
                # Accel_x: mean=2, range [-10, 12], low variation for natural bell curve
                acc_add = self.rng.normal(2, 1.0, 3)  # Mean=2, low variance for natural bell curve
                # Accel_y: mean=0, range [-5, 5], low variation
                acc_add[1] = self.rng.normal(0, 0.5, 1)  # Mean=0, range [-5, 5]
                # Accel_z: mean=-9, range [-15, 0], medium variation
                acc_add[2] = self.rng.normal(0, 1.0, 1)  # Z variation around 0

                # Most rotation in 27-40, very sparse in 40-65 (max 65°/s)
                if self.rng.uniform() < 0.98:
                    rotation_rate = self.rng.uniform(27, 40)  # Most rotations (tăng từ 97% lên 98%)
                else:
                    rotation_rate = self.rng.uniform(40, 65)  # Very sparse high rotations (max 65)
                gyr_add = self.rng.normal(0, 1.8, 3)  # Reduced variance for smoother distribution
                gyr_add[2] += rotation_rate  # Yaw dominant
                
            elif state == 3:  # moving - driving/repositioning
                # Accel_x: mean=2, range [-10, 12], low variation for natural bell curve
                acc_add = self.rng.normal(2, 1.2, 3)  # Mean=2, low variance for natural bell curve
                # Accel_y: mean=0, range [-5, 5], low variation
                acc_add[1] = self.rng.normal(0, 0.5, 1)  # Mean=0, range [-5, 5]
                # Accel_z: mean=-9, range [-15, 0], medium variation
                acc_add[2] = self.rng.normal(0, 1.2, 1)  # Z variation around 0
                gyr_add = self.rng.uniform(3, 27) * self.rng.normal(0, 0.8, 3)  # In plateau range 3-27, smoother
                
            else:
                acc_add = np.zeros(3)
                gyr_add = np.zeros(3)
                
            acc[i] = g_body + acc_add
            gyr[i] = gyr_add
            
        return acc, gyr
    
    def _add_noise_and_artifacts(
        self,
        acc: np.ndarray,
        gyr: np.ndarray,
        active_mask: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise, bias drift, outliers, and apply clipping"""
        N = len(acc)
        
        # White noise (excavator: very low acc variance for natural bell curve)
        acc_noise_sigma = 0.05  # m/s² (very low for natural bell curve)
        gyr_noise_sigma = 0.005  # deg/s (slightly higher for smoother transitions)
        
        for i in range(N):
            if active_mask[i]:
                acc[i] += self.rng.normal(0, acc_noise_sigma, 3)
                gyr[i] += self.rng.normal(0, gyr_noise_sigma, 3)
        
        # Slow bias drift (mean-reverting Ornstein-Uhlenbeck process)
        bias_acc = np.zeros((N, 3))
        bias_gyr = np.zeros((N, 3))
        
        # OU process parameters (excavator sensor drift)
        theta_acc = 0.01  # Mean reversion rate
        theta_gyr = 0.01
        sigma_acc = 0.02 * 9.81  # ~0.02g noise (excavator vibration)
        sigma_gyr = 0.012  # ~0.012 deg/s noise (slightly higher for smoother transitions)
        
        for i in range(1, N):
            # Mean-reverting: dx = -theta * x * dt + sigma * dW
            bias_acc[i] = bias_acc[i-1] - theta_acc * bias_acc[i-1] * dt + \
                         self.rng.normal(0, sigma_acc * np.sqrt(dt), 3)
            bias_gyr[i] = bias_gyr[i-1] - theta_gyr * bias_gyr[i-1] * dt + \
                         self.rng.normal(0, sigma_gyr * np.sqrt(dt), 3)
            
        # Apply bias only during active periods
        for i in range(N):
            if active_mask[i]:
                acc[i] += bias_acc[i]
                gyr[i] += bias_gyr[i]
        
        # Sparse outliers (<0.1%)
        outlier_prob = 0.0008
        for i in range(N):
            if active_mask[i] and self.rng.random() < outlier_prob:
                gyr[i] += self.rng.uniform(40, 60) * self.rng.choice([-1, 1], 3)
        
        # Clipping (max 65°/s for gyro)
        acc = np.clip(acc, -20, 20)
        gyr = np.clip(gyr, -65, 65)

        # Additional clipping for gyro to ensure max 65°/s
        gyro_norm = np.sqrt(np.sum(gyr**2, axis=1))
        for i in range(N):
            if gyro_norm[i] > 65:
                scale = 65 / gyro_norm[i]
                gyr[i] *= scale

        # For excavator: create natural bell curve (6-16) without any hard clipping
        # Remove all clipping to avoid spikes at edges - let natural distribution form
        # Only apply very gentle bounds to prevent extreme outliers
        for i in range(N):
            if active_mask[i]:
                acc_norm_val = np.linalg.norm(acc[i])
                if acc_norm_val < 5.0:
                    # Very gentle scaling only for extreme low values
                    scale = 5.0 / acc_norm_val
                    acc[i] = acc[i] * scale
                elif acc_norm_val > 18.0:
                    # Very gentle scaling only for extreme high values
                    scale = 18.0 / acc_norm_val
                    acc[i] = acc[i] * scale
        
        return acc, gyr
    
    def _build_dataframe(
        self,
        t_utc: np.ndarray,
        t_local: np.ndarray,
        acc: np.ndarray,
        gyr: np.ndarray,
        acc_norm: np.ndarray,
        gyro_norm: np.ndarray,
        N: int
    ) -> pd.DataFrame:
        """Build final dataframe with all required columns"""
        
        # Generate random _id (24-char hex)
        ids = [self._random_hex24() for _ in range(N)]
        
        # Format timestamps
        timestamp_utc = [self._format_timestamp_utc(t) for t in t_utc]
        updated_at = [self._format_updated_at(t) for t in t_utc]
        created_at = timestamp_utc.copy()
        timestamp_local = [self._format_timestamp_local(t) for t in t_local]
        timestamp_vn = [self._format_timestamp_vn(t) for t in t_local]
        
        # Round individual components first, then compute norms from rounded values
        acc_rounded = np.round(acc, 2)
        gyr_rounded = np.round(gyr, 2)
        
        # Recompute norms from rounded values for consistency
        acc_norm_final = np.sqrt(np.sum(acc_rounded**2, axis=1))
        gyro_norm_final = np.sqrt(np.sum(gyr_rounded**2, axis=1))
        
        # Build dataframe
        df = pd.DataFrame({
            '_id': ids,
            'mac': ['ffff00000000'] * N,
            'Accel_x': acc_rounded[:, 0],
            'Accel_y': acc_rounded[:, 1],
            'Accel_z': acc_rounded[:, 2],
            'Gyro_x': gyr_rounded[:, 0],
            'Gyro_y': gyr_rounded[:, 1],
            'Gyro_z': gyr_rounded[:, 2],
            'Timestamp': timestamp_utc,
            'updated_at': updated_at,
            'created_at': created_at,
            '__v': [0] * N,
            'Timestamp_local': timestamp_local,
            'Timestamp_vn': timestamp_vn,
            'acc_norm': np.round(acc_norm_final, 2),
            'gyro_norm': np.round(gyro_norm_final, 2)
        })
        
        return df
    
    def _random_hex24(self) -> str:
        """Generate random 24-character hex string (lowercase)"""
        return ''.join(self.rng.choice(list('0123456789abcdef')) for _ in range(24))
    
    def _format_timestamp_utc(self, t: pd.Timestamp) -> str:
        """Format: 2025-08-04 00:07:06.479000+00:00"""
        return t.strftime('%Y-%m-%d %H:%M:%S.%f') + '+00:00'
    
    def _format_updated_at(self, t: pd.Timestamp) -> str:
        """Format: 2025-08-04T00:07:06.479Z (milliseconds)"""
        return t.strftime('%Y-%m-%dT%H:%M:%S.') + f'{t.microsecond // 1000:03d}Z'
    
    def _format_timestamp_local(self, t: pd.Timestamp) -> str:
        """Format: 2025-08-04 07:07:06.479000+07:00"""
        return t.strftime('%Y-%m-%d %H:%M:%S.%f+07:00')
    
    def _format_timestamp_vn(self, t: pd.Timestamp) -> str:
        """Format: 2025-08-04 07:07:06.479000 (no timezone)"""
        return t.strftime('%Y-%m-%d %H:%M:%S.%f')


def main():
    parser = argparse.ArgumentParser(
        description='IMU Synth Generator - Generate synthetic IMU data'
    )
    parser.add_argument(
        '--start_utc',
        type=str,
        required=True,
        help='Start time in UTC (e.g., "2025-08-04 00:07:00+00:00")'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=1,
        help='Number of days to generate (default: 1)'
    )
    parser.add_argument(
        '--mean_dt',
        type=float,
        default=0.30,
        help='Mean sampling interval in seconds (default: 0.30)'
    )
    parser.add_argument(
        '--jitter',
        type=float,
        default=0.02,
        help='Jitter amplitude for sampling interval (default: 0.02)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--calibrate_from',
        type=str,
        default=None,
        help='Path to 5-row CSV for calibration'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (default: stdout)'
    )
    
    args = parser.parse_args()
    
    # Generate data
    generator = IMUSynthGenerator(seed=args.seed)
    df = generator.generate(
        start_utc=args.start_utc,
        days=args.days,
        mean_dt=args.mean_dt,
        jitter=args.jitter,
        calibrate_from_head5_path=args.calibrate_from
    )
    
    # Output
    if args.output:
        df.to_csv(args.output, index=False, na_rep='NaN')
        print(f"Generated {len(df)} samples to {args.output}", file=sys.stderr)
    else:
        df.to_csv(sys.stdout, index=False, na_rep='NaN')
    
    # Print statistics to stderr
    active_data = df[df['Accel_x'].notna()]
    if len(active_data) > 0:
        print(f"\n=== Statistics (working hours only) ===", file=sys.stderr)
        print(f"Active samples: {len(active_data)}", file=sys.stderr)
        print(f"acc_norm median: {active_data['acc_norm'].median():.2f} m/s²", file=sys.stderr)
        print(f"acc_norm range: [{active_data['acc_norm'].min():.2f}, {active_data['acc_norm'].max():.2f}]", file=sys.stderr)
        print(f"gyro_norm median: {active_data['gyro_norm'].median():.2f} °/s", file=sys.stderr)
        print(f"gyro_norm 95th percentile: {active_data['gyro_norm'].quantile(0.95):.2f} °/s", file=sys.stderr)
        print(f"gyro_norm max: {active_data['gyro_norm'].max():.2f} °/s", file=sys.stderr)


if __name__ == '__main__':
    main()

