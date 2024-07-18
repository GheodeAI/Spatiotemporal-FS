import cdsapi
import numpy as np


def download_data():
    for y in np.arange(2022, 2023):
        c = cdsapi.Client()

        # fmt: off
        c.retrieve(
            "reanalysis-era5-single-levels",  # for downloading z500: 'reanalysis-era5-pressure-levels' and indicate 'pressure_level': '500'
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': 'sea_surface_temperature',
                'year': str(y),
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                90, -180, -90,
                180,
                ],
                'grid': [0.5, 0.5],
            },
            f"./sst/data_sst_{y}.nc",
        )
