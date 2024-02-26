from calc_extrinsic_rgb import calc_extrinsic as calc_extrinsic_rgb
from calc_extrinsic_depth import calc_extrinsic as calc_extrinsic_depth


if __name__ == '__main__':
    print("Calculating extrinsic parameters for RGB cameras...")
    calc_extrinsic_rgb()

    print("Calculating extrinsic parameters for depth cameras...")
    calc_extrinsic_depth()
