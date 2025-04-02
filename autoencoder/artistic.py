import pickle
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap


def extract_lonlat(fpath):
    filename = fpath / Path('latlon.file')
    with open(filename, 'rb') as latlon_file:
        lonlat = pickle.load(latlon_file, encoding="latin1")
    return lonlat

def grid_to_lat_lon(X, Y):
    Lat_min, Lat_max = 37.5, 55.4
    Lon_min, Lon_max = -12.0, 16.0
    n_lat, n_lon = 717, 1121
    Lat = Lat_min + Y * (Lat_max - Lat_min) / n_lat
    Lon = Lon_min + X * (Lon_max - Lon_min) / n_lon
    return Lat, Lon

def get_boundaries(zone):
    zone_idx = {"NO": (230, 300),
                "SO": (250, 150),
                "SE": (540, 120),
                "NE": (460, 300),
                "C": (350, 220),
                "SE_for_GAN": (595, 198),
                "SE_GAN_extend": (540, 120),
                "SE_for_GAN_terrestrial": (500, 180),
                "AROME_all": (0, 0)}
    assert zone in zone_idx.keys(), f'{zone} is not in <zone_idx.keys()>: {zone_idx.keys()}'
    return zone_idx[zone]

class CanvasHolder():
    def __init__(self, zone, nb_lon, nb_lat, fpath, bound_num=1):
        self.X_min, self.Y_min = get_boundaries(zone)
        self.nb_lon, self.nb_lat = nb_lon, nb_lat
        self.lonlat = extract_lonlat(fpath)
        self.coords = [
            self.lonlat[0][self.Y_min:(self.Y_min + nb_lat), self.X_min:(self.X_min + nb_lon)],
            self.lonlat[1][self.Y_min:(self.Y_min + nb_lat), self.X_min:(self.X_min + nb_lon)]
        ]
        self.proj0 = ccrs.Stereographic(central_latitude=46.7, central_longitude=2.0)
        self.proj_plot = ccrs.PlateCarree()
        self.axes_class = (GeoAxes, dict(map_projection=self.proj0))
        shapefile_path = Path(__file__).parent / 'departement' / 'ne_10m_admin_1_states_provinces.shp'
        self.department_borders = cfeature.ShapelyFeature(
            Reader(shapefile_path).geometries(),
            ccrs.PlateCarree(),
            linewidth=0.1,
            edgecolor='black',
            facecolor='none'
        )
        colors = [
            "white", "#63006e", "#0000ff", "#00b2ff", "#00ffff", "#08dfd6", "#1cb8a5",
            "#6ba530", "#ffff00", "#ffd800", "#ffa500", "#ff0000", "#991407", "#ff00ff", "#a4ff00"
        ]
        if bound_num == 1:
            bounds = np.array([0, 0.5, 1, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 1000])
            bounds = np.array([0, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 3, 5, 10, 30, 70, 100, 150, 200, 1000])
        else:
            bounds = np.array([0, 0.2, 0.5, 1, 2, 3, 5, 7, 9, 12, 15, 20])
        self.cmapRR = ListedColormap(colors[:len(bounds) - 1], name="from_list", N=None)
        self.norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds))

    def project(self, padX=(5, 15), padY=(5, 5), ax=None):
        if ax is None:
            ax = plt.axes(projection=self.proj0)
        Lat_min, Lon_min = grid_to_lat_lon(self.X_min - padX[0], self.Y_min - padY[0])
        Lat_max, Lon_max = grid_to_lat_lon(self.X_min + self.nb_lon + padX[1], self.Y_min + self.nb_lat + padY[1])
        lon_bor = [Lon_min, Lon_max]
        lat_bor = [Lat_min, Lat_max]
        lon_lat_1 = self.proj0.transform_point(lon_bor[0], lat_bor[0], ccrs.PlateCarree())
        lon_lat_2 = self.proj0.transform_point(lon_bor[1], lat_bor[1], ccrs.PlateCarree())
        lon_bor = [lon_lat_1[0], lon_lat_2[0]]
        lat_bor = [lon_lat_1[1], lon_lat_2[1]]
        borders = lon_bor + lat_bor
        ax.set_extent(borders, self.proj0)
        return ax

    
    def plot_data_normal(self, rr_map, plot_dir, pic_name):
        fig, ax = plt.subplots(1, 1, figsize=(10, 14), subplot_kw={'projection': self.proj0})
        ax = self.project(ax=ax)

        ax.pcolormesh(self.coords[0], self.coords[1], rr_map, cmap=self.cmapRR, alpha=1, transform=self.proj_plot, norm=self.norm)
        ax.set_title(pic_name)
        ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        ax.add_feature(cfeature.BORDERS.with_scale('10m'))
        ax.add_feature(self.department_borders)
        sm = ScalarMappable(cmap=self.cmapRR, norm=self.norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.94, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_label('mm/h', fontsize='10')
        cbar.ax.tick_params(labelsize='8')
        fig.subplots_adjust(bottom=0.005, top=0.96, left=0, right=0.95, wspace=0.1, hspace=0.2)
        plt.savefig(Path(plot_dir) / pic_name, dpi=400)
        plt.close()
