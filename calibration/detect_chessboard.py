from detect_chessboard_rgb import detect_chessboards as detect_chessboards_rgb
from calibration.detect_chessboard_depth import detect_chessboards as detect_chessboards_depth


if __name__ == '__main__':
    print("Detecting chessboards in RGB images...")
    detect_chessboards_rgb()

    print("Detecting chessboards in infrared images...")
    detect_chessboards_depth()
