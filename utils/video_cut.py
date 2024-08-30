import cv2

from tqdm import tqdm


video_path = "/home/hamid/Documents/phd/footefield/data/Aug 23 real/cam5.mp4"
video_path_output = "/home/hamid/Documents/phd/footefield/data/Aug 23 real/cam5_cut.mp4"
start_frame = 702
end_frame = 702 + 500

if __name__ == "__main__":
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        video_path_output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        5,
        (1920, 1080))

    bar = tqdm(range(end_frame - start_frame))
    for _ in bar:
        ret, frame = cap.read()

        # frame = cv2.resize(frame, (720, 480))

        # cv2.imshow("frame", frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break

        writer.write(frame)
