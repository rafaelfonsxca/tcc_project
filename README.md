# tcc_project

Projeto de Visão Computacional para encontrar logos em videos e realizar o corte automaticamente.

Projeto Final do Curso de Sistemas de Informação.

### Como executar:

Na pasta 'main':

    Configurar o caminho do vídeo, da logotipo alvo e o caminho para salvar o vídeo cortado.
    Se necessário, passar o parametro 'resize' para True.

    Exemplo:
    detector = LogoDetector(video_path='../videos/15s-canto-esquerdo-pequeno.mp4',
                            logo_path='../target/logo_target.png',
                            path_to_save_video='videos_cortados/video_cortado_test_sift2.mp4',
                            resize=True)

    Após isso no terminal ´python app.py´
