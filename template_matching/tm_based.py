import cv2
import json
import os
import time


def detect_and_select_logo(video_path, output_dir="kernel"):
    # Carregar o vídeo
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        return None

    # Verificar se o diretório de saída existe, caso contrário, criar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_idx = 0
    logo_filepath = None

    while True:
        ret, frame = video.read()
        if not ret:
            print("Fim do vídeo ou erro ao ler o vídeo.")
            break

        # Mostrar o frame atual
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(30) & 0xFF

        # Pressionar 'p' para pausar o vídeo e selecionar a logo
        if key == ord("p"):
            # Usar selectROI para selecionar a logo manualmente
            roi = cv2.selectROI(
                "Selecione a logo", frame, fromCenter=False, showCrosshair=True
            )

            # Recortar a área selecionada
            if roi is not None:
                x, y, w, h = roi
                selected_logo = frame[int(y) : int(y + h), int(x) : int(x + w)]

                # Gerar nome para o arquivo da logo
                logo_filename = f"logo_frame_{frame_idx}.png"
                logo_filepath = os.path.join(output_dir, logo_filename)

                # Salvar a logo
                cv2.imwrite(logo_filepath, selected_logo)
                print(f"Logo salva em: {logo_filepath}")
                break

        # Pressionar 'q' para sair do vídeo
        if key == ord("q"):
            break

        frame_idx += 1

    # Fechar vídeo e janelas
    video.release()
    cv2.destroyAllWindows()

    return logo_filepath


def _seconds_to_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


# Lista de métodos
METHODS = [
    "TM_SQDIFF",
    "TM_SQDIFF_NORMED",
    "TM_CCORR",
    "TM_CCORR_NORMED",
    "TM_CCOEFF",
    "TM_CCOEFF_NORMED",
]


def detect_logo_in_video(video_path, logo_path, threshold=0.8, methods: list = METHODS):

    # Carregar o vídeo e o logotipo (template)
    video = cv2.VideoCapture(video_path)
    logo = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)

    if not video.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # logo_height, logo_width = logo.shape[:2]
    # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for method_name in methods:
        time.sleep(5)
        start = time.time()
        # Obter o método pelo nome usando getattr no cv2
        method = getattr(cv2, method_name)
        # Reiniciar o vídeo no início
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fps = video.get(cv2.CAP_PROP_FPS)

        logo_found = False
        logo_start_time = None
        logo_end_time = None
        logo_events = []

        frame_idx = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Converter o frame para escala de cinza para fazer a correspondência
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Aplicar template matching
            res = cv2.matchTemplate(gray_frame, logo, method)
            min_val, max_val, _, _ = cv2.minMaxLoc(res)

            match_val = (
                min_val if method_name in ["TM_SQDIFF", "TM_SQDIFF_NORMED"] else max_val
            )
            confidence = (
                1 - min_val
                if method_name in ["TM_SQDIFF", "TM_SQDIFF_NORMED"]
                else max_val
            )

            # Se a correspondência for maior que o threshold, consideramos que a logo foi encontrada
            if confidence >= threshold:
                if not logo_found:
                    logo_found = True
                    logo_start_time = frame_idx / fps
            else:
                if logo_found:
                    logo_found = False
                    logo_end_time = frame_idx / fps
                    logo_events.append(
                        {
                            "start": _seconds_to_timestamp(logo_start_time),
                            "end": _seconds_to_timestamp(logo_end_time),
                        }
                    )

            frame_idx += 1

        # Verificar se a logo estava visível no final do vídeo
        if logo_found:
            logo_end_time = frame_idx / fps
            logo_events.append(
                {
                    "start": _seconds_to_timestamp(logo_start_time),
                    "end": _seconds_to_timestamp(logo_end_time),
                }
            )

        # Exportar os eventos para JSON
        with open(f"direito_{method_name}_results.json", "w") as json_file:
            json.dump(logo_events, json_file, indent=4)

        print(
            f"Processamento {method_name} concluído em {(time.time() - start), 'seconds'}."
        )

    # Fechar o vídeo
    video.release()


if __name__ == "__main__":
    video = "/Users/rafael.fonseca/Documents/Faculdade 2024.2/MONO II/desenvolvimento/tcc/videos/15s-canto-direito.mp4"
    # kernel = detect_and_select_logo(video)
    logo_target = "/Users/rafael.fonseca/Documents/Faculdade 2024.2/MONO II/desenvolvimento/tcc/target/logo_target.png"
    detect_logo_in_video(video, logo_target, threshold=0.8)
