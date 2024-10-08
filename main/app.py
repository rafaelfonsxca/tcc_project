import cv2
import subprocess
import time

class LogoDetector:
    def __init__(self, video_path, logo_path, path_to_save_video, threshold=0.6, min_match_count=10, ratio_test=0.7, resize=False):
        self.video_path = video_path
        self.logo_path = logo_path
        self.threshold = threshold
        self.min_match_count = min_match_count
        self.ratio_test = ratio_test
        self.resize = resize
        self.path_to_save_video = path_to_save_video

    def _convert_seconds_to_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def tm_ccoeff_normed_based(self):
        cap = cv2.VideoCapture(self.video_path)
        logo = cv2.imread(self.logo_path, 0)  # reading logo target in gray scale

        if not cap.isOpened():
            print("Erro ao abrir o vídeo.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)

        detection_started = False
        logo_timestamps = []
        start_time = None
        end_time = None

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray_frame, logo, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val >= self.threshold:
                if not detection_started:
                    start_time = frame_index / fps
                    detection_started = True
            else:
                if detection_started:
                    end_time = frame_index / fps
                    logo_timestamps.append(
                        {
                            "start": self._convert_seconds_to_timestamp(start_time),
                            "end": self._convert_seconds_to_timestamp(end_time),
                        }
                    )
                    detection_started = False

            frame_index += 1

        if detection_started:
            end_time = frame_index / fps
            logo_timestamps.append(
                {
                    "start": self._convert_seconds_to_timestamp(start_time),
                    "end": self._convert_seconds_to_timestamp(end_time),
                }
            )

        cap.release()
        return logo_timestamps

    def sift_based(self):
        logo_image = cv2.imread(self.logo_path, cv2.IMREAD_GRAYSCALE)
        if logo_image is None:
            raise IOError("Não foi possível carregar a imagem da logotipo.")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError("Não foi possível abrir o vídeo.")

        sift = cv2.SIFT_create()
        _, des_logo = sift.detectAndCompute(logo_image, None)
        if des_logo is None:
            raise ValueError("Não foi possível encontrar descritores na logotipo.")

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        logo_present = False
        start_time = None
        end_time = None
        detections = []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            raise ValueError("FPS do vídeo é 0, verifique o arquivo de vídeo.")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, des_frame = sift.detectAndCompute(frame_gray, None)

            if des_frame is not None:
                try:
                    matches = flann.knnMatch(des_logo, des_frame, k=2)
                except cv2.error as e:
                    print(f"Erro durante a correspondência de características: {e}")
                    matches = []

                good_matches = []
                for m_n in matches:
                    if len(m_n) != 2:
                        continue
                    m, n = m_n
                    if m.distance < self.ratio_test * n.distance:
                        good_matches.append(m)

                if len(good_matches) > self.min_match_count:
                    if not logo_present:
                        logo_present = True
                        start_time = frame_count / fps
                else:
                    if logo_present:
                        logo_present = False
                        end_time = frame_count / fps
                        detections.append({
                            "start": self._convert_seconds_to_timestamp(start_time),
                            "end": self._convert_seconds_to_timestamp(end_time)
                        })
            else:
                if logo_present:
                    logo_present = False
                    end_time = frame_count / fps
                    detections.append({
                        "start": self._convert_seconds_to_timestamp(start_time),
                        "end": self._convert_seconds_to_timestamp(end_time)
                    })

            frame_count += 1

        if logo_present:
            end_time = frame_count / fps
            detections.append({
                "start": self._convert_seconds_to_timestamp(start_time),
                "end": self._convert_seconds_to_timestamp(end_time)
            })

        cap.release()
        return detections
    
    def cut(self, input_video, start_time, end_time, output_video):
        """
        Corta um vídeo de acordo com os timestamps inicial e final.

        Args:
        - input_video (str): Caminho do vídeo de entrada.
        - start_time (str): Timestamp inicial no formato 'hh:mm:ss'.
        - end_time (str): Timestamp final no formato 'hh:mm:ss'.
        - output_video (str): Caminho do vídeo de saída.

        Returns:
        - None
        """
        # Comando ffmpeg para cortar o vídeo
        command = [
            'ffmpeg', 
            '-i', input_video,   # Arquivo de entrada
            '-ss', start_time,   # Tempo inicial
            '-to', end_time,     # Tempo final
            '-c', 'copy',        # Copia sem recodificar
            output_video         # Arquivo de saída
        ]
        
        # Executa o comando
        subprocess.run(command, check=True)
        print(f"Video target was cuted and saved in {output_video}")
    
    def main(self):
        start = time.time()
        if not self.resize:
            print('Start detection with TM_CCOEFF_NORMED from Template_Matching.')
            detections = self.tm_ccoeff_normed_based()
            self.cut(self.video_path, detections[0]['start'], detections[0]['end'], self.path_to_save_video)
        else:
            pass
        print(f'System execution time: {time.time() - start} seconds')

    

if __name__ == '__main__':
    detector = LogoDetector(video_path='../videos/15s-canto-direito.mp4',
                            logo_path='../target/logo_target.png',
                            path_to_save_video='videos_cortados/video_cortado_test_2.mp4')
    detector.main()