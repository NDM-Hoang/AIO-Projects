# 🚜 IMU Synthetic Data Generator - Excavator

Bộ công cụ tạo dữ liệu IMU tổng hợp chuyên nghiệp cho máy xúc với các đặc tính thực tế hoàn hảo.

## 📋 Tổng quan

Generator này tạo ra dữ liệu IMU tổng hợp (3-axis accelerometer + 3-axis gyroscope) mô phỏng chính xác hoạt động của máy xúc trong giờ làm việc thực tế. Được tối ưu hóa với các đặc tính kỹ thuật cao cấp:

### ✨ Đặc điểm nổi bật
- **Accelerometer**: Mean và range chính xác cho từng trục với phân bố tự nhiên
- **acc_norm**: Hình chuông hoàn hảo (5-18 m/s²) không có spikes ở biên
- **Gyro histogram**: Phân bố đặc biệt với 3 peaks + plateau đặc trưng của máy xúc
- **Thời gian**: Chỉ tạo dữ liệu trong giờ làm việc thực tế với khả năng tùy chỉnh hoàn toàn

## 🚀 Cài đặt & Thiết lập

### 1. Tạo môi trường ảo
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

## 📖 Hướng dẫn sử dụng

### 1. Command Line (Đơn giản nhất)
```bash
# Tạo dữ liệu cho ngày mặc định (2025-08-04) → 20250804_sensor_data.csv
python generate_excavator_data.py

# Tạo dữ liệu cho ngày cụ thể → 20250815_sensor_data.csv
python generate_excavator_data.py --date 2025-08-15

# Tạo dữ liệu nhiều ngày với tham số tùy chỉnh
python generate_excavator_data.py --date 2025-08-06 --days 3 --seed 123
```

### 2. Python API (Linh hoạt hơn)
```python
from imu_synth_generator import IMUSynthGenerator

# Khởi tạo generator
generator = IMUSynthGenerator(seed=42)

# Tạo dữ liệu cơ bản
df = generator.generate('2025-08-04 00:07:00+00:00', days=1)
df.to_csv('basic_data.csv', index=False)

# Tạo dữ liệu với giờ làm việc tùy chỉnh
df = generator.generate(
    start_utc='2025-08-04 00:07:00+00:00',
    days=2,
    morning_start='06:00',
    morning_end='08:30',
    afternoon_start='14:00',
    afternoon_end='18:00'
)
df.to_csv('custom_hours.csv', index=False)
```

## 📊 Đặc tính kỹ thuật

### Accelerometer (3 trục)
| Trục | Mean | Range | Đặc điểm |
|------|------|-------|----------|
| **Accel_x** | ≈ 2 m/s² | [-10, 12] | Dao động cao, mô phỏng chuyển động ngang |
| **Accel_y** | ≈ 0 m/s² | [-5, 5] | Ổn định tương đối |
| **Accel_z** | ≈ -9 m/s² | [-15, 0] | Hướng xuống, bao gồm trọng lực |

### acc_norm (Vector tổng hợp)
- **Hình dạng**: Bell curve hoàn hảo tự nhiên
- **Range**: [5.0, 18.0] m/s²
- **Mean**: ≈ 10.3 m/s²
- **Đặc điểm**: Không có spikes ở biên, phân bố mượt mà

### Gyroscope Histogram (Đặc trưng máy xúc)
| Range | Đặc điểm | Ý nghĩa |
|-------|----------|---------|
| **0°/s** | ~12,000 samples | Máy nghỉ, động cơ chạy không tải |
| **1°/s** | ~800 samples | Spike nhỏ - chuyển động nhẹ |
| **2°/s** | ~400 samples | Spike trung bình - điều chỉnh vị trí |
| **3-27°/s** | Plateau ~300/bin | Vùng làm việc chính - đào/xoay |
| **27-40°/s** | Drop nhanh | Chuyển tiếp nhanh |
| **40-65°/s** | Sparse | Xoay nhanh/điều kiện đặc biệt |

### Thời gian làm việc
- **Buổi sáng**: 07:00 - 09:45 (2h45m) ⏰
- **Buổi chiều**: 13:45 - 17:00 (3h15m) ⏰
- **Tổng**: 6 giờ/ngày = 21,600 samples 📊
- **Tùy chỉnh**: Hoàn toàn có thể thay đổi giờ bắt đầu/kết thúc

## 🔧 Tham số tùy chỉnh

### Command Line Arguments
```bash
python generate_excavator_data.py [OPTIONS]

Các tùy chọn:
  --date DATE           Ngày tạo dữ liệu (YYYY-MM-DD, mặc định: 2025-08-04)
  --days DAYS           Số ngày tạo dữ liệu (mặc định: 1)
  --seed SEED           Seed ngẫu nhiên (mặc định: 42)
  --output OUTPUT       Tên file xuất (mặc định: tự động theo ngày YYYYMMDD_sensor_data.csv)
  --morning_start TIME  Giờ bắt đầu sáng (HH:MM, mặc định: 07:00)
  --morning_end TIME    Giờ kết thúc sáng (HH:MM, mặc định: 09:45)
  --afternoon_start TIME Giờ bắt đầu chiều (HH:MM, mặc định: 13:45)
  --afternoon_end TIME  Giờ kết thúc chiều (HH:MM, mặc định: 17:00)
```

### Python API Parameters
```python
generator.generate(
    start_utc='2025-08-04 00:07:00+00:00',  # Thời gian bắt đầu (UTC)
    days=1,                                   # Số ngày
    morning_start='07:00',                    # Giờ bắt đầu sáng
    morning_end='09:45',                      # Giờ kết thúc sáng
    afternoon_start='13:45',                  # Giờ bắt đầu chiều
    afternoon_end='17:00'                     # Giờ kết thúc chiều
)
```

## 📈 Ví dụ thực tế

### Output mẫu (1 ngày)
```
🚜 Excavator IMU Data Generator
============================================================
Date:       2025-08-04
Days:       1
Seed:       42
Output:     20250804_sensor_data.csv
Morning:   07:00 - 09:45
Afternoon: 13:45 - 17:00

⏳ Generating data...
💾 Saving to 20250804_sensor_data.csv...

✅ Generation Complete!
============================================================
Total samples:  21,600
Active samples: 21,600 (100.0%)

📊 Statistics (working hours):
  acc_norm:  median=10.05 m/s², range=[5.0, 16.7]
  gyro_norm: median=0.15 °/s, max=65 °/s

⏰ Working Hours (VN time):
  Morning:   07:00 - 09:45
  Afternoon: 13:45 - 17:00
```

### Format CSV đầu ra
```csv
Timestamp_utc,Timestamp_vn,Accel_x,Accel_y,Accel_z,Gyro_x,Gyro_y,Gyro_z,acc_norm,gyro_norm
2025-08-04 00:07:00+00:00,2025-08-04 07:07:00+07:00,1.83,0.28,-9.82,0.01,0.02,0.00,10.38,0.02
2025-08-04 00:07:01+00:00,2025-08-04 07:07:01+07:00,1.85,0.31,-9.79,0.03,-0.01,0.01,10.41,0.03
...
```

## 📁 Cấu trúc dự án

```
IMU-Synth-Generator/
├── imu_synth_generator.py      # Core generator class với đầy đủ tính năng
├── generate_excavator_data.py  # Script đơn giản để sử dụng command line
├── requirements.txt           # Dependencies cần thiết
└── README.md                 # Tài liệu hướng dẫn này
```

## 🎯 Ứng dụng thực tế

- **🤖 Machine Learning**: Training data chất lượng cao cho mô hình phân loại hoạt động máy xúc
- **🧪 Sensor Simulation**: Mô phỏng dữ liệu IMU chân thực cho testing và validation
- **📊 Research**: Nghiên cứu về hoạt động và hiệu suất của máy xúc
- **🔄 Data Augmentation**: Tăng cường dataset thực tế với dữ liệu tổng hợp đa dạng

## 🔍 Troubleshooting

### Các lỗi thường gặp
1. **ModuleNotFoundError**: `pip install -r requirements.txt`
2. **Permission denied**: Kiểm tra quyền ghi thư mục
3. **Memory error**: Giảm `--days` hoặc tăng RAM hệ thống

### Kiểm tra dữ liệu
```python
import pandas as pd
import matplotlib.pyplot as plt

# Đọc và kiểm tra dữ liệu
df = pd.read_csv('20250804_sensor_data.csv')

# Thống kê cơ bản
print(f"Samples: {len(df):,}")
print(f"acc_norm: mean={df['acc_norm'].mean():.2f}, range=[{df['acc_norm'].min():.1f}, {df['acc_norm'].max():.1f}]")

# Vẽ histogram kiểm tra
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(df['acc_norm'], bins=50, alpha=0.7)
plt.title('acc_norm Distribution')
plt.subplot(1, 2, 2)
plt.hist(df['gyro_norm'], bins=50, alpha=0.7)
plt.title('gyro_norm Distribution')
plt.tight_layout()
plt.show()
```

## 📝 Lịch sử phát triển

### v3.0 (Current - Hoàn thiện)
- ✅ **Natural bell curve**: acc_norm không còn spikes ở biên
- ✅ **Tên file tự động**: `YYYYMMDD_sensor_data.csv`
- ✅ **Tham số ngày tháng**: `--date` thay vì `--start_utc`
- ✅ **Giờ làm việc tùy chỉnh**: Đầy đủ tham số cho cả sáng và chiều
- ✅ **Documentation hoàn chỉnh**: README với ví dụ chi tiết

### v2.1
- Excavator simulation hoàn thiện
- Gyro histogram với 3 peaks đặc trưng
- Working hours tùy chỉnh

### v2.0
- Chuyển từ human activity sang excavator
- Gyro histogram refinement

### v1.0
- Basic IMU generation cho hoạt động con người

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 Giấy phép

MIT License - Chi tiết xem file LICENSE

## 📞 Liên hệ & Hỗ trợ

Nếu gặp vấn đề hoặc cần hỗ trợ:
- Tạo issue trên GitHub
- Liên hệ trực tiếp với nhóm phát triển

---

**🎯 Sẵn sàng tạo dữ liệu IMU máy xúc chân thực nhất!** 🚜

*Chạy `python generate_excavator_data.py --help` để xem tất cả tùy chọn*