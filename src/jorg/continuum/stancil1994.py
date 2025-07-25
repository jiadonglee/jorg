"""
Stancil1994 molecular ion cross-sections for H₂⁺ and He₂⁺.

This module contains transcriptions of the tables from
Stancil 1994 (https://ui.adsabs.harvard.edu/abs/1994ApJ...430..360S/abstract), who calculated ff
and bf absorption coefficients for H₂⁺ and He₂⁺.

It also contains pre-constructed interpolation objects.
"""

import jax.numpy as jnp
from jax import jit
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from typing import Tuple


class Stancil1994Data:
    """Container for Stancil 1994 molecular ion cross-section data."""
    
    def __init__(self):
        """Initialize Stancil 1994 data with tabulated values."""
        
        # Temperature arrays for equilibrium constants
        self.Ts_He2plus = np.array([4200, 6300, 8400, 12600, 16800, 25200, 33600, 50400])
        self.K_He2plus_vals = 1e21 * np.array([0.3606, 2.6330, 7.1409, 21.097, 39.842, 87.960, 147.41, 293.57])
        
        self.Ts_H2plus = np.array([3150, 4200, 5040, 6300, 8400, 12600, 16800, 25200])
        self.K_H2plus_vals = 1e19 * np.array([0.9600, 9.7683, 29.997, 89.599, 265.32, 845.01, 1685.3, 4289.5])
        
        # Wavelength arrays (converted from nm to Å)
        self.λs_H2plus_ff = 10.0 * np.array([
            70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220,
            230, 240, 250, 260, 270, 280, 290, 295, 300, 350, 400, 450, 500, 600, 700,
            800, 900, 1000, 2000, 3000, 4000, 5000, 11000, 15000, 20000
        ])
        
        self.λs_He2plus_ff = 10.0 * np.array([
            70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
            220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 600, 700,
            800, 900, 1000, 2000, 3000, 4000, 5000, 11000, 15000, 20000
        ])
        
        # For bf absorption, wavelengths are the same for H₂⁺ and He₂⁺
        self.λs_bf = 10.0 * np.array([
            50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210,
            220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 600, 700, 800,
            900, 1000, 2000, 3000, 4500, 5000, 10000, 15000, 20000
        ])
        
        # Cross-section tables - He₂⁺ free-free (units: cm⁻⁵)
        self.σ_He2plus_ff_table = np.array([
            [7.08e-3, 6.05e-3, 5.45e-3, 4.78e-3, 4.41e-3, 3.99e-3, 3.75e-3, 3.50e-3],
            [0.0103, 8.76e-3, 7.84e-3, 6.79e-3, 6.20e-3, 5.52e-3, 5.15e-3, 4.74e-3],
            [0.0134, 0.0113, 0.0101, 8.66e-3, 7.87e-3, 6.97e-3, 6.48e-3, 5.91e-3],
            [0.0165, 0.0139, 0.0124, 0.0106, 9.65e-3, 8.52e-3, 7.89e-3, 7.19e-3],
            [0.0196, 0.0165, 0.0147, 0.0126, 0.0114, 0.0101, 9.30e-3, 8.47e-3],
            [0.0226, 0.0190, 0.0169, 0.0145, 0.0131, 0.0116, 0.0107, 9.72e-3],
            [0.0259, 0.0217, 0.0193, 0.0166, 0.0150, 0.0132, 0.0122, 0.0111],
            [0.0288, 0.0242, 0.0215, 0.0185, 0.0167, 0.0147, 0.0136, 0.0124],
            [0.0317, 0.0267, 0.0237, 0.0204, 0.0185, 0.0163, 0.0151, 0.0137],
            [0.0347, 0.0292, 0.0260, 0.0223, 0.0203, 0.0179, 0.0165, 0.0151],
            [0.0377, 0.0317, 0.0283, 0.0243, 0.0220, 0.0195, 0.0180, 0.0164],
            [0.0405, 0.0341, 0.0304, 0.0262, 0.0237, 0.0210, 0.0195, 0.0178],
            [0.0433, 0.0365, 0.0326, 0.0280, 0.0254, 0.0225, 0.0209, 0.0191],
            [0.0460, 0.0388, 0.0346, 0.0298, 0.0271, 0.0240, 0.0223, 0.0204],
            [0.0487, 0.0411, 0.0367, 0.0316, 0.0288, 0.0255, 0.0237, 0.0217],
            [0.0514, 0.0434, 0.0387, 0.0335, 0.0304, 0.0270, 0.0251, 0.0230],
            [0.0540, 0.0457, 0.0408, 0.0352, 0.0321, 0.0285, 0.0266, 0.0244],
            [0.0567, 0.0479, 0.0429, 0.0371, 0.0338, 0.0301, 0.0280, 0.0257],
            [0.0593, 0.0502, 0.0449, 0.0389, 0.0354, 0.0316, 0.0294, 0.0271],
            [0.0619, 0.0524, 0.0469, 0.0407, 0.0371, 0.0331, 0.0308, 0.0284],
            [0.0645, 0.0546, 0.0489, 0.0424, 0.0387, 0.0346, 0.0323, 0.0297],
            [0.0669, 0.0567, 0.0508, 0.0441, 0.0403, 0.0360, 0.0336, 0.0310],
            [0.0693, 0.0587, 0.0527, 0.0457, 0.0418, 0.0374, 0.0350, 0.0323],
            [0.0716, 0.0607, 0.0545, 0.0474, 0.0433, 0.0388, 0.0363, 0.0335],
            [0.0825, 0.0702, 0.0632, 0.0551, 0.0506, 0.0455, 0.0427, 0.0396],
            [0.0934, 0.0798, 0.0719, 0.0630, 0.0580, 0.0524, 0.0494, 0.0460],
            [0.1036, 0.0887, 0.0802, 0.0706, 0.0651, 0.0591, 0.0558, 0.0522],
            [0.1122, 0.0964, 0.0873, 0.0771, 0.0714, 0.0650, 0.0615, 0.0577],
            [0.1280, 0.1106, 0.1006, 0.0894, 0.0831, 0.0762, 0.0724, 0.0683],
            [0.1446, 0.1255, 0.1146, 0.1024, 0.0956, 0.0882, 0.0841, 0.0797],
            [0.1573, 0.1371, 0.1257, 0.1129, 0.1058, 0.0980, 0.0938, 0.0892],
            [0.1673, 0.1465, 0.1347, 0.1215, 0.1143, 0.1063, 0.1020, 0.0973],
            [0.1787, 0.1570, 0.1448, 0.1312, 0.1237, 0.1155, 0.1111, 0.1064],
            [0.2615, 0.2361, 0.2221, 0.2067, 0.1984, 0.1894, 0.1847, 0.1796],
            [0.3078, 0.2825, 0.2687, 0.2537, 0.2456, 0.2371, 0.2325, 0.2278],
            [0.3556, 0.3301, 0.3162, 0.3014, 0.2934, 0.2850, 0.2806, 0.2760],
            [0.4175, 0.3909, 0.3766, 0.3612, 0.3531, 0.3445, 0.3400, 0.3353],
            [0.5835, 0.5607, 0.5487, 0.5360, 0.5294, 0.5225, 0.5190, 0.5153],
            [0.6620, 0.6406, 0.6293, 0.6176, 0.6114, 0.6051, 0.6018, 0.5985],
            [0.7563, 0.7359, 0.7252, 0.7140, 0.7083, 0.7023, 0.6992, 0.6961]
        ])
        
        # Cross-section tables - He₂⁺ bound-free (units: cm⁻⁵)
        self.σ_He2plus_bf_table = np.array([
            [5.99e-6, 1.15e-5, 1.50e-5, 1.86e-5, 2.62e-5, 2.18e-5, 2.25e-5, 2.31e-5],
            [4.31e-2, 8.49e-3, 0.0112, 0.0140, 0.0153, 0.0165, 0.0171, 0.0176],
            [0.0523, 0.0816, 0.0973, 0.1108, 0.1166, 0.1213, 0.1232, 0.1249],
            [0.2503, 0.2883, 0.2956, 0.2927, 0.2876, 0.2809, 0.2770, 0.2729],
            [0.6587, 0.6169, 0.5716, 0.5169, 0.4869, 0.4584, 0.4446, 0.4313],
            [1.2197, 1.0054, 0.8757, 0.7493, 0.6879, 0.6333, 0.6081, 0.5842],
            [1.7949, 1.3801, 1.1630, 0.9667, 0.8758, 0.7973, 0.7617, 0.7285],
            [2.2739, 1.6955, 1.4083, 1.1557, 1.0409, 0.9426, 0.8984, 0.8573],
            [2.5948, 1.9282, 1.5999, 1.3121, 1.1816, 1.0701, 1.0200, 0.9734],
            [2.7699, 2.0876, 1.7438, 1.4385, 1.2986, 1.1784, 1.1241, 1.0735],
            [2.8293, 2.1868, 1.8478, 1.5389, 1.3950, 1.2699, 1.2129, 1.1596],
            [2.7978, 2.2354, 1.9185, 1.6198, 1.4775, 1.3523, 1.2948, 1.2407],
            [2.7012, 2.2420, 1.9582, 1.6787, 1.5418, 1.4194, 1.3625, 1.3087],
            [2.5652, 2.2185, 1.9747, 1.7208, 1.5924, 1.4754, 1.4203, 1.3678],
            [2.4108, 2.1766, 1.9764, 1.7532, 1.6360, 1.5269, 1.4750, 1.4250],
            [2.2515, 2.1226, 1.9663, 1.7752, 1.6702, 1.5701, 1.5217, 1.4748],
            [2.0956, 2.0615, 1.9468, 1.7873, 1.6944, 1.6034, 1.5586, 1.5146],
            [1.9478, 1.9976, 1.9221, 1.7934, 1.7127, 1.6310, 1.5899, 1.5492],
            [1.8099, 1.9334, 1.8946, 1.7958, 1.7275, 1.6553, 1.6182, 1.5809],
            [1.6822, 1.8694, 1.8646, 1.7945, 1.7382, 1.6756, 1.6425, 1.6087],
            [1.5643, 1.8060, 1.8322, 1.7894, 1.7449, 1.6919, 1.6628, 1.6326],
            [1.4557, 1.7438, 1.7985, 1.7817, 1.7488, 1.7054, 1.6805, 1.6541],
            [1.3561, 1.6834, 1.7641, 1.7720, 1.7506, 1.7168, 1.6962, 1.6736],
            [1.2648, 1.6249, 1.7291, 1.7601, 1.7497, 1.7253, 1.7088, 1.6901],
            [1.1815, 1.5685, 1.6935, 1.7458, 1.7457, 1.7302, 1.7177, 1.7026],
            [1.1056, 1.5144, 1.6579, 1.7297, 1.7391, 1.7320, 1.7234, 1.7119],
            [0.8192, 1.2864, 1.4950, 1.6420, 1.6905, 1.7188, 1.7266, 1.7304],
            [0.6390, 1.1210, 1.3675, 1.5665, 1.6448, 1.7017, 1.7233, 1.7406],
            [0.5156, 0.9900, 1.2562, 1.4893, 1.5890, 1.6678, 1.7003, 1.7283],
            [0.4267, 0.8818, 1.1552, 1.4080, 1.5221, 1.6161, 1.6566, 1.6927],
            [0.3145, 0.7275, 1.0010, 1.2736, 1.4051, 1.5195, 1.5711, 1.6187],
            [0.2506, 0.6309, 0.9016, 1.1870, 1.3314, 1.4619, 1.5226, 1.5799],
            [0.2079, 0.5578, 0.8202, 1.1078, 1.2583, 1.3977, 1.4638, 1.5271],
            [0.1772, 0.4991, 0.7498, 1.0327, 1.1842, 1.3269, 1.3955, 1.4618],
            [0.1551, 0.4541, 0.6941, 0.9710, 1.1219, 1.2660, 1.3360, 1.4041],
            [0.0716, 0.2458, 0.4036, 0.6028, 0.7194, 0.8366, 0.8958, 0.9549],
            [0.0454, 0.1634, 0.2743, 0.4207, 0.5041, 0.5919, 0.6368, 0.6820],
            [0.0344, 0.1268, 0.2151, 0.3334, 0.4015, 0.4738, 0.5110, 0.5486],
            [0.0290, 0.1088, 0.1861, 0.2908, 0.3516, 0.4165, 0.4501, 0.4841],
            [0.0134, 0.0517, 0.0900, 0.1429, 0.1742, 0.2080, 0.2257, 0.2437],
            [0.0104, 0.0408, 0.0711, 0.1134, 0.1384, 0.1656, 0.1798, 0.1943],
            [8.88e-3, 0.0349, 0.0611, 0.0976, 0.1194, 0.1430, 0.1554, 0.1681]
        ])
        
        # Cross-section tables - H₂⁺ bound-free (units: cm⁻⁵)
        self.σ_H2plus_bf_table = np.array([
            [7.34e-5, 1.43e-4, 2.04e-4, 2.87e-4, 3.89e-4, 4.97e-4, 5.49e-4, 5.98e-4],
            [0.0100, 0.0150, 0.0186, 0.0230, 0.0276, 0.0319, 0.0337, 0.0353],
            [0.1676, 0.1965, 0.2105, 0.2215, 0.2266, 0.2246, 0.2211, 0.2163],
            [0.8477, 0.8199, 0.7797, 0.7183, 0.6376, 0.5477, 0.5037, 0.4622],
            [2.1113, 1.8166, 1.6135, 1.3823, 1.1403, 0.9157, 0.8176, 0.7313],
            [3.4427, 2.8069, 2.4213, 2.0136, 1.6137, 1.2616, 1.1129, 0.9845],
            [4.3470, 3.5155, 3.0224, 2.5070, 2.0062, 1.5685, 1.3846, 1.2262],
            [4.6981, 3.8841, 3.3806, 2.8402, 2.3019, 1.8203, 1.6147, 1.4358],
            [4.6169, 3.9763, 3.5361, 3.0358, 2.5120, 2.0239, 1.8096, 1.6202],
            [4.2811, 3.8840, 3.5476, 3.1272, 2.6535, 2.1858, 1.9729, 1.7809],
            [3.8331, 3.6850, 3.4660, 3.1434, 2.7388, 2.3082, 2.1033, 1.9138],
            [3.3624, 3.4344, 3.3301, 3.1098, 2.7840, 2.4020, 2.2106, 2.0287],
            [2.9167, 3.1670, 3.1662, 3.0438, 2.7985, 2.4707, 2.2958, 2.1244],
            [2.5172, 2.9031, 2.9909, 2.9573, 2.7898, 2.5175, 2.3610, 2.2019],
            [2.1697, 2.6538, 2.8149, 2.8600, 2.7657, 2.5495, 2.4128, 2.2682],
            [1.8726, 2.4243, 2.6444, 2.7573, 2.7300, 2.5682, 2.4516, 2.3222],
            [1.6208, 2.2160, 2.4821, 2.6522, 2.6843, 2.5725, 2.4748, 2.3597],
            [1.4085, 2.0289, 2.3297, 2.5473, 2.6314, 2.5653, 2.4852, 2.3839],
            [1.2295, 1.8617, 2.1883, 2.4453, 2.5755, 2.5517, 2.4889, 2.4012],
            [1.0785, 1.7129, 2.0581, 2.3479, 2.5188, 2.5347, 2.4885, 2.4144],
            [0.9508, 1.5804, 1.9387, 2.2553, 2.4622, 2.5145, 2.4840, 2.4230],
            [0.8424, 1.4623, 1.8293, 2.1680, 2.4063, 2.4920, 2.4762, 2.4277],
            [0.7501, 1.3568, 1.7296, 2.0865, 2.3530, 2.4701, 2.4687, 2.4327],
            [0.6712, 1.2626, 1.6392, 2.0118, 2.3045, 2.4525, 2.4660, 2.4433],
            [0.6033, 1.1783, 1.5577, 1.9443, 2.2620, 2.4415, 2.4707, 2.4626],
            [0.5448, 1.1029, 1.4843, 1.8834, 2.2256, 2.4369, 2.4825, 2.4901],
            [0.3477, 0.8233, 1.2037, 1.6442, 2.0832, 2.4343, 2.5567, 2.6400],
            [0.2412, 0.6469, 1.0069, 1.4533, 1.9363, 2.3659, 2.5349, 2.6658],
            [0.1782, 0.5283, 0.8605, 1.2951, 1.7920, 2.2630, 2.4604, 2.6222],
            [0.1389, 0.4487, 0.7547, 1.1780, 1.6859, 2.1963, 2.4228, 2.6179],
            [0.0927, 0.3396, 0.6120, 1.0109, 1.5255, 2.0874, 2.3564, 2.6031],
            [0.0685, 0.2749, 0.5175, 0.8894, 1.3923, 1.9700, 2.2587, 2.5323],
            [0.0542, 0.2335, 0.4534, 0.8033, 1.2940, 1.8800, 2.1824, 2.4760],
            [0.0448, 0.2041, 0.4046, 0.7334, 1.2067, 1.7876, 2.0941, 2.3965],
            [0.0382, 0.1813, 0.3635, 0.6701, 1.1194, 1.6816, 1.9827, 2.2828],
            [0.0159, 0.0901, 0.1982, 0.3951, 0.7108, 1.1448, 1.3953, 1.6590],
            [0.0100, 0.0596, 0.1325, 0.2699, 0.4954, 0.8132, 1.0003, 1.1999],
            [6.88e-3, 0.0425, 0.0962, 0.1994, 0.3723, 0.6216, 0.7710, 0.9325],
            [6.41e-3, 0.0400, 0.0908, 0.1889, 0.3540, 0.5932, 0.7371, 0.8932],
            [3.56e-3, 0.0229, 0.0526, 0.1110, 0.2109, 0.3582, 0.4479, 0.5463],
            [2.50e-3, 0.0161, 0.0373, 0.0790, 0.1506, 0.2567, 0.3216, 0.3929],
            [1.90e-3, 0.0123, 0.0286, 0.0607, 0.1161, 0.1982, 0.2487, 0.3042]
        ])
        
        # Cross-section tables - H₂⁺ free-free (units: cm⁻⁵)
        self.σ_H2plus_ff_table = np.array([
            [0.0174, 0.0154, 0.0142, 0.0130, 0.0116, 0.0100, 9.10e-3, 8.08e-3],
            [0.0280, 0.0246, 0.0227, 0.0207, 0.0184, 0.0158, 0.0143, 0.0126],
            [0.0394, 0.0346, 0.0319, 0.0290, 0.0257, 0.0220, 0.0199, 0.0175],
            [0.0514, 0.0451, 0.0416, 0.0378, 0.0336, 0.0287, 0.0259, 0.0227],
            [0.0640, 0.0562, 0.0519, 0.0471, 0.0418, 0.0357, 0.0322, 0.0283],
            [0.0770, 0.0676, 0.0624, 0.0567, 0.0504, 0.0431, 0.0389, 0.0341],
            [0.0903, 0.0794, 0.0733, 0.0666, 0.0592, 0.0506, 0.0457, 0.0401],
            [0.1040, 0.0914, 0.0843, 0.0767, 0.0682, 0.0584, 0.0527, 0.0464],
            [0.1177, 0.1035, 0.0956, 0.0869, 0.0773, 0.0663, 0.0599, 0.0527],
            [0.1317, 0.1158, 0.1070, 0.0973, 0.0866, 0.0743, 0.0672, 0.0592],
            [0.1456, 0.1281, 0.1184, 0.1078, 0.0960, 0.0824, 0.0746, 0.0658],
            [0.1597, 0.1405, 0.1299, 0.1183, 0.1054, 0.0906, 0.0821, 0.0725],
            [0.1737, 0.1530, 0.1414, 0.1288, 0.1149, 0.0988, 0.0896, 0.0793],
            [0.1877, 0.1654, 0.1529, 0.1394, 0.1243, 0.1071, 0.0972, 0.0861],
            [0.2017, 0.1777, 0.1644, 0.1499, 0.1338, 0.1154, 0.1048, 0.0930],
            [0.2156, 0.1901, 0.1759, 0.1605, 0.1433, 0.1237, 0.1125, 0.0998],
            [0.2294, 0.2023, 0.1873, 0.1709, 0.1527, 0.1319, 0.1201, 0.1068],
            [0.2431, 0.2145, 0.1987, 0.1814, 0.1622, 0.1402, 0.1277, 0.1137],
            [0.2568, 0.2266, 0.2099, 0.1917, 0.1716, 0.1485, 0.1354, 0.1206],
            [0.2703, 0.2387, 0.2211, 0.2020, 0.1809, 0.1567, 0.1430, 0.1276],
            [0.2836, 0.2506, 0.2322, 0.2123, 0.1902, 0.1649, 0.1506, 0.1345],
            [0.2969, 0.2624, 0.2433, 0.2225, 0.1994, 0.1731, 0.1582, 0.1414],
            [0.3100, 0.2741, 0.2542, 0.2325, 0.2086, 0.1812, 0.1657, 0.1483],
            [0.3165, 0.2799, 0.2596, 0.2376, 0.2131, 0.1853, 0.1695, 0.1518],
            [0.3230, 0.2857, 0.2650, 0.2425, 0.2177, 0.1893, 0.1732, 0.1552],
            [0.3858, 0.3419, 0.3176, 0.2913, 0.2621, 0.2290, 0.2103, 0.1894],
            [0.4451, 0.3952, 0.3677, 0.3378, 0.3048, 0.2674, 0.2463, 0.2228],
            [0.5011, 0.4457, 0.4152, 0.3821, 0.3456, 0.3044, 0.2812, 0.2555],
            [0.5539, 0.4935, 0.4603, 0.4243, 0.3848, 0.3401, 0.3150, 0.2873],
            [0.6511, 0.5821, 0.5442, 0.5032, 0.4583, 0.4078, 0.3795, 0.3484],
            [0.7388, 0.6625, 0.6207, 0.5756, 0.5262, 0.4710, 0.4402, 0.4064],
            [0.8186, 0.7362, 0.6911, 0.6425, 0.5895, 0.5303, 0.4975, 0.4615],
            [0.8918, 0.8042, 0.7563, 0.7049, 0.6488, 0.5864, 0.5518, 0.5141],
            [0.9596, 0.8675, 0.8173, 0.7633, 0.7047, 0.6395, 0.6036, 0.5644],
            [1.4600, 1.3450, 1.2830, 1.2170, 1.1460, 1.0680, 1.0260, 0.9815],
            [1.8050, 1.6820, 1.6170, 1.5470, 1.4740, 1.3940, 1.3510, 1.3060],
            [2.2000, 2.0750, 2.0090, 1.9390, 1.8650, 1.7860, 1.7450, 1.7000],
            [2.3130, 2.1880, 2.1220, 2.0520, 1.9790, 1.9010, 1.8590, 1.8160],
            [3.1800, 3.0620, 3.0000, 2.9360, 2.8690, 2.7980, 2.7610, 2.7230],
            [3.8110, 3.7000, 3.6420, 3.5820, 3.5200, 3.4560, 3.4220, 3.3870],
            [4.3230, 4.2180, 4.1640, 4.1080, 4.0500, 3.9900, 3.9580, 3.9260]
        ])
        
        # Convert cross-sections to CGS units
        self.σ_He2plus_ff_table *= 1e-39  # Convert to cm⁻⁵ from table units
        self.σ_He2plus_bf_table *= 1e-18  # Convert to cm⁻⁵ from table units
        self.σ_H2plus_ff_table *= 1e-39   # Convert to cm⁻⁵ from table units
        self.σ_H2plus_bf_table *= 1e-18   # Convert to cm⁻⁵ from table units
        
        # Create interpolators
        self._create_interpolators()
    
    def _create_interpolators(self):
        """Create 2D interpolators for cross-sections."""
        # He₂⁺ interpolators
        self.σ_He2plus_ff_interpolator = RegularGridInterpolator(
            (self.λs_He2plus_ff, self.Ts_He2plus), 
            self.σ_He2plus_ff_table,
            bounds_error=False,
            fill_value=None,
            method='linear'
        )
        
        self.σ_He2plus_bf_interpolator = RegularGridInterpolator(
            (self.λs_bf, self.Ts_He2plus), 
            self.σ_He2plus_bf_table,
            bounds_error=False,
            fill_value=None,
            method='linear'
        )
        
        # H₂⁺ interpolators
        self.σ_H2plus_ff_interpolator = RegularGridInterpolator(
            (self.λs_H2plus_ff, self.Ts_H2plus), 
            self.σ_H2plus_ff_table,
            bounds_error=False,
            fill_value=None,
            method='linear'
        )
        
        self.σ_H2plus_bf_interpolator = RegularGridInterpolator(
            (self.λs_bf, self.Ts_H2plus), 
            self.σ_H2plus_bf_table,
            bounds_error=False,
            fill_value=None,
            method='linear'
        )
        
        # Equilibrium constant interpolators
        self.K_He2plus_interpolator = RegularGridInterpolator(
            (self.Ts_He2plus,), 
            self.K_He2plus_vals,
            bounds_error=False,
            fill_value=None,
            method='linear'
        )
        
        self.K_H2plus_interpolator = RegularGridInterpolator(
            (self.Ts_H2plus,), 
            self.K_H2plus_vals,
            bounds_error=False,
            fill_value=None,
            method='linear'
        )
    
    def he2plus_ff_cross_section(self, wavelength: float, temperature: float) -> float:
        """
        Get He₂⁺ free-free cross-section.
        
        Args:
            wavelength: Wavelength in Å
            temperature: Temperature in K
            
        Returns:
            Cross-section in cm⁻⁵
        """
        result = self.σ_He2plus_ff_interpolator(np.array([wavelength, temperature]))[0]
        return max(0.0, result)  # Ensure non-negative
    
    def he2plus_bf_cross_section(self, wavelength: float, temperature: float) -> float:
        """
        Get He₂⁺ bound-free cross-section.
        
        Args:
            wavelength: Wavelength in Å
            temperature: Temperature in K
            
        Returns:
            Cross-section in cm⁻⁵
        """
        result = self.σ_He2plus_bf_interpolator(np.array([wavelength, temperature]))[0]
        return max(0.0, result)  # Ensure non-negative
    
    def h2plus_ff_cross_section(self, wavelength: float, temperature: float) -> float:
        """
        Get H₂⁺ free-free cross-section.
        
        Args:
            wavelength: Wavelength in Å
            temperature: Temperature in K
            
        Returns:
            Cross-section in cm⁻⁵
        """
        result = self.σ_H2plus_ff_interpolator(np.array([wavelength, temperature]))[0]
        return max(0.0, result)  # Ensure non-negative
    
    def h2plus_bf_cross_section(self, wavelength: float, temperature: float) -> float:
        """
        Get H₂⁺ bound-free cross-section.
        
        Args:
            wavelength: Wavelength in Å
            temperature: Temperature in K
            
        Returns:
            Cross-section in cm⁻⁵
        """
        result = self.σ_H2plus_bf_interpolator(np.array([wavelength, temperature]))[0]
        return max(0.0, result)  # Ensure non-negative
    
    def he2plus_equilibrium_constant(self, temperature: float) -> float:
        """
        Get He₂⁺ equilibrium constant.
        
        Args:
            temperature: Temperature in K
            
        Returns:
            Equilibrium constant in cm⁻³
        """
        return self.K_He2plus_interpolator(np.array([temperature]))[0]
    
    def h2plus_equilibrium_constant(self, temperature: float) -> float:
        """
        Get H₂⁺ equilibrium constant.
        
        Args:
            temperature: Temperature in K
            
        Returns:
            Equilibrium constant in cm⁻³
        """
        return self.K_H2plus_interpolator(np.array([temperature]))[0]


# Global instance
_STANCIL_DATA = Stancil1994Data()


def get_he2plus_ff_cross_section(wavelength: float, temperature: float) -> float:
    """Get He₂⁺ free-free cross-section."""
    return _STANCIL_DATA.he2plus_ff_cross_section(wavelength, temperature)


def get_he2plus_bf_cross_section(wavelength: float, temperature: float) -> float:
    """Get He₂⁺ bound-free cross-section."""
    return _STANCIL_DATA.he2plus_bf_cross_section(wavelength, temperature)


def get_h2plus_ff_cross_section(wavelength: float, temperature: float) -> float:
    """Get H₂⁺ free-free cross-section."""
    return _STANCIL_DATA.h2plus_ff_cross_section(wavelength, temperature)


def get_h2plus_bf_cross_section(wavelength: float, temperature: float) -> float:
    """Get H₂⁺ bound-free cross-section."""
    return _STANCIL_DATA.h2plus_bf_cross_section(wavelength, temperature)


def get_he2plus_equilibrium_constant(temperature: float) -> float:
    """Get He₂⁺ equilibrium constant."""
    return _STANCIL_DATA.he2plus_equilibrium_constant(temperature)


def get_h2plus_equilibrium_constant(temperature: float) -> float:
    """Get H₂⁺ equilibrium constant."""
    return _STANCIL_DATA.h2plus_equilibrium_constant(temperature)


def get_molecular_cross_sections(wavelength: float, temperature: float) -> Tuple[float, float, float, float]:
    """
    Get all molecular ion cross-sections at once.
    
    Args:
        wavelength: Wavelength in Å
        temperature: Temperature in K
        
    Returns:
        Tuple of (He₂⁺_ff, He₂⁺_bf, H₂⁺_ff, H₂⁺_bf) cross-sections in cm⁻⁵
    """
    return (
        get_he2plus_ff_cross_section(wavelength, temperature),
        get_he2plus_bf_cross_section(wavelength, temperature),
        get_h2plus_ff_cross_section(wavelength, temperature),
        get_h2plus_bf_cross_section(wavelength, temperature)
    )


def get_equilibrium_constants(temperature: float) -> Tuple[float, float]:
    """
    Get all equilibrium constants at once.
    
    Args:
        temperature: Temperature in K
        
    Returns:
        Tuple of (He₂⁺, H₂⁺) equilibrium constants in cm⁻³
    """
    return (
        get_he2plus_equilibrium_constant(temperature),
        get_h2plus_equilibrium_constant(temperature)
    )