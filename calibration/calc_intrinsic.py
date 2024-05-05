from calc_intrinsic_rgb import calc_intrinsic as calc_intrinsic_rgb
from calc_intrinsic_depth import calc_intrinsic as calc_intrinsic_depth


if __name__ == '__main__':
    print("Calculating intrinsic parameters for RGB cameras...")
    calc_intrinsic_rgb()

    print("Calculating intrinsic parameters for depth cameras...")
    calc_intrinsic_depth()
