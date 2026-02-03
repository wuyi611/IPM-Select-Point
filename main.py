import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os


class IPMPointCollector:
    def __init__(self, rtsp_url, xml_path, display_width=960):
        self.rtsp_url = rtsp_url
        self.display_width = display_width
        self.mtx, self.dist, self.new_mtx = self.load_xml_params(xml_path)
        self.cap = cv2.VideoCapture(rtsp_url)
        self.image_orig = None
        self.display_scale = 1.0

    def load_xml_params(self, xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            def get_data(node_name, shape):
                container = root.find(node_name)
                data = [float(container.find(f'data{i}').text) for i in range(shape[0] * shape[1])]
                return np.array(data).reshape(shape)

            return get_data('camera_matrix', (3, 3)), get_data('camera_distortion', (1, 5)), get_data(
                'new_camera_matrix', (3, 3))
        except Exception as e:
            print(f"XML解析失败: {e}")
            exit()

    def get_undistorted_snapshot(self):
        print("正在连接 RTSP 并获取最新帧...")
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.rtsp_url)

        for _ in range(5):
            self.cap.grab()

        ret, frame = self.cap.read()
        if not ret: return False

        self.image_orig = cv2.undistort(frame, self.mtx, self.dist, None, self.new_mtx)
        h, w = self.image_orig.shape[:2]
        self.display_scale = self.display_width / w
        self.image_display = cv2.resize(self.image_orig, (self.display_width, int(h * self.display_scale)))
        return True

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = x / self.display_scale
            orig_y = y / self.display_scale
            param['temp_pts'].append([orig_x, orig_y])
            cv2.circle(param['img'], (x, y), 5, (0, 255, 0), -1)
            cv2.putText(param['img'], f"Click_{len(param['temp_pts'])}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def get_intersect(self, a, b, c, d):
        denom = (d[1] - c[1]) * (b[0] - a[0]) - (d[0] - c[0]) * (b[1] - a[1])
        if abs(denom) < 1e-6: return None
        ua = ((d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0])) / denom
        return [a[0] + ua * (b[0] - a[0]), a[1] + ua * (b[1] - a[1])]

    def run(self):
        print("\n模式: [1] 虚拟点推算+补全  [2] 直接选取4点")
        mode = input("模式选择: ")

        if not self.get_undistorted_snapshot():
            print("错误：无法连接流或读取图像")
            return

        final_points = [None] * 4
        cv2.namedWindow("Calibration")
        context = {'temp_pts': [], 'img': self.image_display.copy()}

        cv2.setMouseCallback("Calibration", self.mouse_callback, context)

        if mode == '1':
            print("\n[阶段 1] 请点击4个点以推算虚拟交点（1-2线，3-4线）")
            while len(context['temp_pts']) < 4:
                cv2.imshow("Calibration", context['img'])
                if cv2.waitKey(20) & 0xFF == 27: return

            v_point = self.get_intersect(context['temp_pts'][0], context['temp_pts'][1],
                                         context['temp_pts'][2], context['temp_pts'][3])

            print(f"\n>>> 推算出的虚拟点原图坐标: {v_point}")
            try:
                idx_input = input("请输入该虚拟点作为第几个点 (1-4): ")
                idx = int(idx_input) - 1
            except ValueError:
                idx = 0
                print("输入无效，默认为第 1 个点")

            if 0 <= idx <= 3:
                final_points[idx] = v_point
            else:
                final_points[0] = v_point

            print(f"\n[阶段 2] 虚拟点已占用位置 {idx + 1}，请继续点击剩下的 3 个点")
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

        # --- 控制台输出 ---
        print("\n" + "=" * 40)
        print("最终获取的四个点坐标 (基于 1080p 原图):")
        for i, pt in enumerate(final_points):
            if pt is not None:
                print(f"Point {i + 1}: [{pt[0]:.4f}, {pt[1]:.4f}]")
        print("=" * 40)

        # --- 保存图像 (全分辨率) ---
        print("\n正在保存文件...")
        save_img = self.image_orig.copy()
        draw_scale = 1.0 / self.display_scale
        radius = int(5 * draw_scale)
        font_scale = 0.6 * draw_scale
        thickness = int(2 * draw_scale)

        for i, pt in enumerate(final_points):
            if pt is not None:
                center = (int(pt[0]), int(pt[1]))
                h, w = save_img.shape[:2]
                if 0 <= center[0] < w and 0 <= center[1] < h:
                    cv2.circle(save_img, center, radius, (0, 255, 0), -1)
                    cv2.putText(save_img, f"P{i + 1}", (center[0] + radius, center[1] - radius),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

        img_save_path = "calib_result_fullres.jpg"
        cv2.imwrite(img_save_path, save_img)
        print(f"[图片] 已保存至: {os.path.abspath(img_save_path)}")

        # --- 新增: 保存坐标到 TXT ---
        txt_save_path = "calib_points.txt"
        with open(txt_save_path, "w") as f:
            for i, pt in enumerate(final_points):
                if pt is not None:
                    # 格式: x, y
                    f.write(f"{pt[0]:.6f}, {pt[1]:.6f}\n")
                else:
                    f.write("None\n")
        print(f"[坐标] 已保存至: {os.path.abspath(txt_save_path)}")

        print("\n任务完成，程序即将退出...")
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    RTSP_URL = "rtsp://192.168.9.83:554/11"
    XML_PATH = "camera_params.xml"
    tool = IPMPointCollector(RTSP_URL, XML_PATH)
    tool.run()