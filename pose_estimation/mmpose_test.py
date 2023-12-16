import cv2
from mmpose.apis import MMPoseInferencer


img_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5/color/color00000.jpg'
image = cv2.imread(img_path)

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer('human')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(image)
for result in result_generator:
    for predictions in result['predictions'][0]:
        keypoints = predictions['keypoints']
        for idx, point in enumerate(keypoints):

            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                image, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)

            cv2.putText(
                image, str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
    cv2.imshow('frame', image)
    if cv2.waitKey(0) == ord('q'):
        break