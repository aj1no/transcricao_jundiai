# transcricao_jundiai
Projeto de reconhecimento de texto cursivo manuscrito (HTR) para documentos históricos de Jundiaí.

Plano de Implementação: Transcrição Histórica de Jundiaí
O objetivo deste projeto é construir um programa em Python usando Machine Learning (Aprendizado de Máquina) focado em Reconhecimento de Texto Manuscrito (HTR) para digitalizar o acervo histórico de Jundiaí (1657 a 1889).

Desafio Principal: Paleografia e Imagens em Alta Resolução
Documentos de 1657 a 1889 apresentam desafios imensos como variação caligráfica, desgaste natural do papel antigo, manchas e tinta enfraquecida. Nossa estratégia será "ensinar" uma IA inteligente, que já sabe ler letras de forma genérica, a se especializar no contexto daquela época através das transcrições que você já tem.

Como o Programa irá funcionar? (Arquitetura)
Vamos dividir a criação do projeto em etapas e criaremos scripts separados de Python para cada uma:

1. Processamento e Recorte da Imagem (Segmentação de Linhas)
A IA não lê páginas enormes de uma vez; ela é feita para ler linhas fragmentadas. Teremos que pegar suas fotos de alta resolução e aplicar filtros de Visão Computacional (ex: OpenCV).

Ação: Um script (processador_imagens.py) para encontrar as linhas manuscritas nas fotos grandes e salvar cada linha como uma imagem fina e separada.
2. O Problema do Alinhamento (O "Treino")
Você tem a foto da página e a transcrição da página. Mas, para treinar a IA matematicamente, precisamos dar na mão dela: "A imagem X (Apenas desta linha)" significa exatamente o "Texto Y".

Ação: Vamos criar uma lógia (preparar_dados.py) para associar os recortes das linhas aos textos que você tem.
3. A Inteligência Artificial (TrOCR)
A Microsoft lançou uma IA de ponta chamada TrOCR (Transformer-based Optical Character Recognition). Ela funciona super bem para caligrafias complexas e cursivas com letras contínuas.

Ação: Usaremos as bibliotecas Transformers e PyTorch em Python (treinar_modelo.py). A IA treinará conectando imagens das caligrafias tortuosas aos textos que lhes pertencem. Os séculos 17 ao 19 costumam utilizar um português diferente (ex: pharmacia, scripto), e o fine-tuning ensinará essa peculiaridade à IA.
4. O Programa de Execução Final (Inferência)
Ação: Entregarei para você um arquivo como transcrever_documento.py. Você aponta para a "foto123.jpg" recém digitalizada, o programa corta as linhas automaticamente, joga todas as linhas na IA treinada, e te retorna um arquivo .txt com toda a documentação transcrita.
User Review Required
IMPORTANT

Precisamos tomar uma decisão arquitetural sobre como as suas transcrições atuais estão organizadas no computador, pois isso determina a dificuldade de avançarmos.

As transcrições que você tem seguem exatamente as mesmas "quebras de linha" de parágrafo que existem fisicamente escrito no papel envelhecido? Ou elas são um bloco de texto corrido sem considerar quando a linha física da tinta acaba?
No seu computador (C:\Users\takem\...), você tem noção da capacidade de hardware dele? Para rodar um grande montante de imagens precisaremos avaliar se você possui alguma placa de vídeo (como uma NVIDIA) ou se rodaremos tudo usando o processador padrão.
