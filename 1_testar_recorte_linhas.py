import cv2
import numpy as np

def testar_recorte_linhas(caminho_imagem):
    print(f"Processando a imagem: {caminho_imagem}...")
    
    # 1. Carregar imagem
    img = cv2.imread(caminho_imagem)
    if img is None:
        print("ERRO: Imagem não encontrada. Verifique o caminho.")
        return
        
    # Diminuir a imagem se for muito grande para vizualização (opcional)
    # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Suavização (remover texturas e manchas finas do papel antigo)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    
    # 3. Binarização Adaptativa 
    #   (Isso é mágico para papéis antigos porque lida bem com luz irregular e bordas escuras)
    thresh = cv2.adaptiveThreshold(
        blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        35, 15
    )
    
    # 4. Dilatação (Juntar as letras soltas em grandes "blocos/faixas" horizontais)
    #   Como a escrita cursiva é horizontal, usamos um retângulo longo e estreito
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # 5. Achar os contornos dessas faixas
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_resultado = img.copy()
    contador_linhas = 0
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Ignorar manchas pequenas (ruídos) ou blocos verticais estranhos
        if w > 100 and h > 20 and h < 200: 
            # Desenha um retângulo vermelho em volta da linha encontrada
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 0, 255), 2)
            contador_linhas += 1
            
    # 6. Salvar o resultado para você ver
    nome_saida = "resultado_teste_linhas.jpg"
    cv2.imwrite(nome_saida, img_resultado)
    print(f"Sucesso! Encontrei aprox. {contador_linhas} blocos que parecem ser linhas.")
    print(f"Abra a imagem '{nome_saida}' para ver como ficou!")

if __name__ == "__main__":
    # COLOQUE O NOME/CAMINHO DA SUA IMAGEM AQUI!
    # Exemplo: testar_recorte_linhas("documento_1663.jpg")
    print("Por favor, edite o código no final do arquivo e coloque o caminho da sua imagem para testar.")
