import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
from datetime import datetime


class IPMPointCollector:
    def __init__(self, rtsp_url, xml_path, display_width=960):
        self.rtsp_url = rtsp_url
        self.display_width = display_width
        self.mtx, self.dist, self.new_mtx = self.load_xml_params(xml_path)
        self.cap = cv2.VideoCapture(rtsp_url)
        self.image_orig = None  # 存储去畸变后的全分辨率原图
        self.display_scale = 1.0

    def load_xml_params(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            def get_data(node_name, shape):
                container = root.find(node_name)
                data = [float(container.find(f'data{i}').text) for i in range(shape[0] * shape[1])]
                return np.array(data).reshape(shape)

            return (get_data('camera_matrix', (3, 3)),
                    get_data('camera_distortion', (1, 5)),
                    get_data('new_camera_matrix', (3, 3)))
        except Exception as e:
            print(f"XML解析失败: {e}")
            exit()

    def get_undistorted_snapshot(self):
        print("正在连接 RTSP 并获取最新帧...")
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.rtsp_url)

        for _ in range(10):  # 连续抓取以排空缓冲区
            self.cap.grab()

        ret, frame = self.cap.read()
        if not ret: return False

        # --- 核心步骤：畸变矫正 ---
        # 得到去畸变后的全分辨率图像
        self.image_orig = cv2.undistort(frame, self.mtx, self.dist, None, self.new_mtx)

        h, w = self.image_orig.shape[:2]
        self.display_scale = self.display_width / w
        self.image_display = cv2.resize(self.image_orig, (self.display_width, int(h * self.display_scale)))
        return True

    def screenshot_mode(self):
        """进入截图预览模式"""
        print("\n" + "=" * 40)
        print(">>> 进入去畸变预览模式 <<<")
        print("操作指令:")
        print("  [S] - 保存当前去畸变截图 (全分辨率)")
        print("  [空格/回车] - 确认并进入下一步(选点)")
        print("  [ESC/Q] - 退出程序")
        print("=" * 40)

        window_name = "Undistorted Preview (Screenshot Mode)"
        cv2.namedWindow(window_name)

        while True:
            cv2.imshow(window_name, self.image_display)
            key = cv2.waitKey(1) & 0xFF

            # 保存截图
            if key == ord('s') or key == ord('S'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"undistorted_snap_{timestamp}.jpg"
                cv2.imwrite(filename, self.image_orig)
                print(f" [已保存] 畸变矫正图: {os.path.abspath(filename)}")

            # 下一步
            elif key == ord(' ') or key == 13:
                print("确认无误，进入选点流程...")
                break

            # 退出
            elif key == 27 or key == ord('q'):
                cv2.destroyAllWindows()
                exit()

        cv2.destroyWindow(window_name)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = x / self.display_scale
            orig_y = y / self.display_scale
            param['temp_pts'].append([orig_x, orig_y])
            cv2.circle(param['img'], (x, y), 5, (0, 255, 0), -1)
            cv2.putText(param['img'], f"P{len(param['temp_pts'])}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def get_intersect(self, a, b, c, d):
        denom = (d[1] - c[1]) * (b[0] - a[0]) - (d[0] - c[0]) * (b[1] - a[1])
        if abs(denom) < 1e-6: return None
        ua = ((d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0])) / denom
        return [a[0] + ua * (b[0] - a[0]), a[1] + ua * (b[1] - a[1])]

    def run(self):
        # 1. 获取图像
        if not self.get_undistorted_snapshot():
            print("错误：无法连接流或读取图像")
            return

        # 2. 截图确认阶段 (新增)
        self.screenshot_mode()

        # 3. 选点模式选择
        print("\n模式选择: [1] 虚拟点推算  [2] 直接选取4点")
        mode = input("请选择: ")

        final_points = [None] * 4
        cv2.namedWindow("Calibration")
        context = {'temp_pts': [], 'img': self.image_display.copy()}
        cv2.setMouseCallback("Calibration", self.mouse_callback, context)

        if mode == '1':
            print("\n[阶段 1] 点击4个点以推算交点 (1-2线, 3-4线)")
            while len(context['temp_pts']) < 4:
                cv2.imshow("Calibration", context['img'])
                if cv2.waitKey(20) & 0xFF == 27: return

            v_point = self.get_intersect(context['temp_pts'][0], context['temp_pts'][1],
                                         context['temp_pts'][2], context['temp_pts'][3])

            print(f">>> 虚拟点坐标: {v_point}")
            idx = int(input("该点序号(1-4): ") or 1) - 1
            final_points[max(0, min(idx, 3))] = v_point

            print(f"[阶段 2] 请点击剩下的 3 个点")
            context['temp_pts'] = []
            while len(context['temp_pts']) < 3:
                cv2.imshow("Calibration", context['img'])
                if cv2.waitKey(20) & 0xFF == 27: return

            click_idx = 0
            for i in range(4):
                if final_points[i] is None:
                    final_points[i] = context['temp_pts'][click_idx]
                    click_idx += 1
        else:
            print("\n[直接选点] 请依次点击 4 个点")
            while len(context['temp_pts']) < 4:
                cv2.imshow("Calibration", context['img'])
                if cv2.waitKey(20) & 0xFF == 27: return
            final_points = context['temp_pts']

        # 4. 保存结果
        print("\n正在保存坐标数据...")
        with open("calib_points.txt", "w") as f:
            for i, pt in enumerate(final_points):
                f.write(f"{pt[0]:.6f}, {pt[1]:.6f}\n")
                print(f"Point {i + 1}: {pt}")

        print("\n任务完成。")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RTSP_URL = "rtsp://192.168.9.83:554/11"
    XML_PATH = "camera_params.xml"
    tool = IPMPointCollector(RTSP_URL, XML_PATH)
    tool.run()