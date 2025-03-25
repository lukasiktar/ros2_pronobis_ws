import rclpy
from rclpy.node import Node
from custom_interfaces.msg import ImageMeta
from cv_bridge import CvBridge
import cv2
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, qos_profile_system_default
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup
import argparse
import torch
import sys
import os
import shutil
from scipy.interpolate import splprep, splev
import numpy as np
sys.path.append("/home/crta-hp-408/PRONOBIS/ros2_pronobis_ws/src/microsegnet_inference/microsegnet_inference")
from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import time

# Node
class MicrosegnetNode(Node):
    def __init__(self):
        super().__init__('microsegnet_node')

        cb_grup = ReentrantCallbackGroup()

        #Ultrasound image subscriber
        self.create_subscription(ImageMeta, '/US_image', self.read_ultrasound_callback, qos_profile=qos_profile_system_default,callback_group=cb_grup)
        self.bridge = CvBridge()
        
        #Process triggers
        self.client_start_sweep=self.create_client(Trigger, "/sweep/start")
        self.client_change_direction=self.create_client(Trigger, "/sweep/edge_trigger",callback_group=cb_grup)

        #Start Trigger init
        # Create a Trigger request to start rotation 
        request = Trigger.Request()
        self.get_logger().info('Start sweep.')                 
        # Send the request asynchronously
        future = self.client_start_sweep.call_async(request)
        # Add a callback to handle the response
        future.add_done_callback(self.response_callback)

        #Microsegnet settings
        self.MODEL_PATH="/home/crta-hp-408/PRONOBIS/MicroSegNet/model/CRTA_MicroSegmentMicroUS224_R50-ViT-B_16_weight4_epo30_bs4_ev02/epoch_29.pth"
        MAIN_DIRECTORY_NAME="/home/crta-hp-408/PRONOBIS/vtk/microsegnet"
        self.INPUT_IMAGES_DIRECTORY=f"{MAIN_DIRECTORY_NAME}/input_images"
        self.OUTPUT_MASKS_DIRECTORY=f"{MAIN_DIRECTORY_NAME}/output_images"
        self.OUTPUT_SEGMENTATIONS_DIRECTORY=f"{MAIN_DIRECTORY_NAME}/output_segmentations"
        import os

        if os.path.exists(MAIN_DIRECTORY_NAME):
            shutil.rmtree(MAIN_DIRECTORY_NAME)
        os.makedirs(MAIN_DIRECTORY_NAME, exist_ok=True)  # Create directory if it doesn't exist
        os.makedirs(self.OUTPUT_MASKS_DIRECTORY, exist_ok=True)  # Create directory if it doesn't exist
        os.makedirs(self.OUTPUT_SEGMENTATIONS_DIRECTORY, exist_ok=True) 
        os.makedirs(self.INPUT_IMAGES_DIRECTORY, exist_ok=True) 

        #Define arguments for MicroSegNet model initialization
        parser = argparse.ArgumentParser()
        parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
        parser.add_argument('--num_classes', type=int,default=1, help='output channel of network')
        parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
        parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
        parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')

        args, unknown = parser.parse_known_args()


        #Define ViT model and load weights
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
        if args.vit_name.find('R50') !=-1:
            config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
        self.net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

        self.net.load_state_dict(torch.load(self.MODEL_PATH), strict=False)
        self.net.eval()
        #Ultrasound roation angle check
        self.angle=None
        self.segmenation_start=False

        #Blank annotation counter - for switching the sweep orientation
        self.blank_images_counter=0
        self.good_images_counter=0
        
        #Sweep recording information
        self.start_sweep_recording=False
        self.stop_sweep_recording=False

        self.switch_rotation=False
        self.future_flag = False

        #Start recording flag
        self.start_recording=False
        self.trigger_counter=0

    #Read the data callback
    def read_ultrasound_callback(self, msg):

        #Ultrasound image
        image = msg.image
        self.cv2_image = self.bridge.imgmsg_to_cv2(image)
        #self.get_logger().info(f'Angle: {self.image_angle:.10f}')

        patch_size=[224, 224]
        x, y = patch_size[0], patch_size[1]

        #Ultrasound image angle
        self.image_angle = msg.angle
        self.get_logger().info(f'Angle: {self.image_angle:.10f}')
        # output_path_original_image= os.path.join(self.INPUT_IMAGES_DIRECTORY, f"original_{self.image_angle:.6f}.png")
        # cv2.imwrite(output_path_original_image, self.cv2_image)

        #Store the image_angle to stored angle in the first step at segmentation start
        if self.segmenation_start == False:
            self.angle = self.image_angle
            self.segmenation_start=True

        #Check if the image_angle is different to the stored angle (if they are the same, it is the same image)
        if self.image_angle != self.angle:
            
            self.angle = self.image_angle
            #print(type(self.angle),flush=True)
            self.image= cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2GRAY)
            
            
            #Resize the tensor
            self.resized_image=cv2.resize(self.image, (224,224))

            #Input to the neural network
            input_nn = torch.from_numpy(self.resized_image).unsqueeze(0).unsqueeze(0).float().cuda()

            with torch.no_grad():
                start_time = time.time()
                #Model outputs
                outputs, _, _, _, cls_output  = self.net(input_nn)
                
                # Apply sigmoid to classification output
                cls_pred = torch.sigmoid(cls_output).squeeze()
                #self.get_logger().info(f"{cls_pred}")
                #Check if the classification predicts an object (you may need to adjust the threshold)
                if cls_pred.item() < 0.99:  
                    # If no object is predicted, create a black mask
                    pred = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
                    self.blank_images_counter+=1
                    self.good_images_counter = 0
                    #If there is 4 frames without annotations
                    if self.blank_images_counter >= 2:
                        
                        if self.switch_rotation==False and not self.future_flag:

                            self.trigger_counter+=1

                            if self.trigger_counter==1:
                                    self.start_recording=True
                            if self.trigger_counter==2:
                                    self.start_recording=False
                                    self.trigger_counter=0

                            # Create a Trigger request to switch rotation 
                            request = Trigger.Request()
                            self.get_logger().info('Switching the sweep direction.')
                            self.future_flag = True
                            
                            # Send the request asynchronously
                            future = self.client_change_direction.call_async(request)

                            # Add a callback to handle the response
                            future.add_done_callback(self.response_callback)
                            self.switch_rotation=True
                            
                    
                    if self.start_sweep_recording==True:
                            self.start_sweep_recording=False       

                else:
                    #Model predictions
                    out = torch.sigmoid(outputs).squeeze()
                    pred = out.cpu().detach().numpy()
                    self.good_images_counter += 1       

                    if self.good_images_counter >=3:
                        self.blank_images_counter=0
                        self.switch_rotation=False

                    if self.start_sweep_recording==False:
                            self.start_sweep_recording=True

                if self.start_recording == True:
                    self.get_logger().info('Recording!')

                    if x != patch_size[0] or y != patch_size[1]:
                        pred = cv2.resize(out, (y, x), interpolation = cv2.INTER_NEAREST)
                    
                    #Create a binary mask from predicitons
                    a = 1.0*(pred>0.5)
                    prediction = a.astype(np.uint8)
                    #Resize to original image (fix - autmate)
                    prediction=cv2.resize(prediction, (self.image.shape[1],self.image.shape[0]))
                    prediction = cv2.normalize(prediction, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    #Find contours on predicted masks (used for visualization)
                    contours, hierarchy = cv2.findContours(prediction,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    #Smooth the contours
                    smoothened = []
                    CURVE_THRESHOLD = 0.018 # Threshold for curvature filtering

                    for contour in contours:
                        
                        # Compute perimeter
                        perimeter = cv2.arcLength(contour, closed=True)

                        # Approximate polygon and check deviation
                        approx = cv2.approxPolyDP(contour, epsilon=CURVE_THRESHOLD * perimeter, closed=True)
                        if len(approx) > 10:  # You can tweak this threshold
                            continue

                        x_1,y_1 = contour.T
                        # Convert from numpy arrays to normal arrays
                        x_1 = x_1.tolist()[0]
                        y_1 = y_1.tolist()[0]
                        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                        tck, u = splprep([x_1,y_1], u=None, s=0.0, k=1, per=1)
                        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                        u_new = np.linspace(u.min(), u.max(), 100)

                        
                        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                        x_new, y_new = splev(u_new, tck, der=0)
                        # Convert it back to numpy format for opencv to be able to display it
                        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
                        smoothened.append(np.asarray(res_array, dtype=np.int32))

                    #Smooth the conoturs and store them
                    prediction[:] = 0  
                    cv2.drawContours(prediction, smoothened, 0, (255,255,255),-1)
                    output_mask_path = os.path.join(self.OUTPUT_MASKS_DIRECTORY, f"slice_{self.image_angle:.6f}.png")

                    cv2.imwrite(output_mask_path, prediction)

                    #Show only the biggest contour
                    cv2.drawContours(self.cv2_image, smoothened, 0, (255,255,255),3)
                    cv2.imshow("Segmentation", self.cv2_image)
                    output_segmentation_path = os.path.join(self.OUTPUT_SEGMENTATIONS_DIRECTORY, f"segmentation_{self.image_angle:.6f}.png")
                    cv2.imwrite(output_segmentation_path, self.cv2_image)

                    output_path_original_image= os.path.join(self.INPUT_IMAGES_DIRECTORY, f"original_{self.image_angle:.6f}.png")
                    cv2.imwrite(output_path_original_image, self.cv2_image)

                    end_time = time.time()
                    inference_time = end_time - start_time
                    self.get_logger().info(f'Inference time: {inference_time:.6f}')
                    #self.get_logger().info(f'Angle: {self.image_angle:.10f}')
                    cv2.waitKey(1)

    def response_callback(self, future):
            self.future_flag = False
            try:
                response = future.result()
                if response.success:
                    self.get_logger().info(f'Service call success: {response.message}')
                else:
                    self.get_logger().warn(f'Service call failed: {response.message}')
            except Exception as e:
                self.get_logger().error(f'Service call failed with exception: {e}')

def main(args=None):
    rclpy.init(args=args)
    microsegnet_node = MicrosegnetNode()
    rclpy.spin(microsegnet_node)
    microsegnet_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
