# pip install opencv-python < Instalar antes de rodar

# CARREGA DEPENDENCIAS
import time
import cv2

# CORES DE CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# Carregando as classes
class_name = []
with open('coco.names', 'r') as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Captura de Video / Use 0 para usar webcam ou coloque o caminho de um video entre ''
cap = cv2.VideoCapture('avenida.mp4')

# Defina a resolução desejada
desired_width = 1300
desired_height = 800

# Configurando a resolução da captura de vídeo
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Carregando Peso da Rede NEURAL > tiny = leve
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

# Parametros da Rede Neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# Leitura dos frames do video
prev_frame = 0  # Frame anterior, usado para mostrar o fps
while True:
    # Captura do frame
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar o frame para a resolução desejada
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Começo da contagem dos MS
    start = time.time()

    # Detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # Fim da Contagem MS
    end = time.time()

    # Percorrer todas as detecções
    for (classid, score, box) in zip(classes, scores, boxes):
        # Gerando cores para as classes
        color = COLORS[int(classid) % len(COLORS)]

        # Pegando o nome da classe pelo ID e o seu Score
        label = f'{class_names[classid]} : {int(score * 100)}%'

        # Desenhando a Box da detecção
        cv2.rectangle(frame, box, color, 2)

        # Escrevendo o nome da classe em cima da box do objeto
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Calculando o tempo que levou para fazer a detecção
    fps_label = 1 / (start - prev_frame)
    prev_frame = start
    fps_label = int(fps_label)
    fps_label = str(fps_label)

    # Escrevendo o FPS na Imagem
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Mostrando a Imagem
    cv2.imshow('detections', frame)

    # Apertando Q no teclado a qualquer momento para fechar o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberação da câmera e destruição de todas as janelas
cap.release()
cv2.destroyAllWindows()
