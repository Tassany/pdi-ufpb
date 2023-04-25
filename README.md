# Processamento Digital de Imagem

O processamento digital de imagens é uma técnica que consiste na mani-
pulação de imagens digitais com o objetivo de melhorar sua qualidade, extrair
informações ou transformá-las de alguma forma.

---

## Projeto

Dividido em 4 arquivos cada um com uma função especifica.

- O arquivo **rgb-yiq.py** é um especifico para a tranformação da imagem de RGB para YIQ, e para a transformação da imagem em negativo.
- O arquivo **mediana.py** especifico para o filtro da mediana, podendo ser regulado o tamanho da mascará MxN 
- O arquivov**histograma.py** faz a expansão de histograma da imagem que resultante do filtro de Sobel
- O arquivo **filtros.py** é responsavel por aplicar o filtro especificado no arquivo **input.txt**, podendo ser ele **Sobel**, **Box-to-Box**, **Emboss**, **Soma**, segue o exemplo de como especificar cada um.

---
## Especificação do input.txt

A primeira linha para dizer as dimenções da mascara, a segunda o nome do filtro, e as seguintes usadas apenas no caso do **Emboss** para especificar a mascará usada. O filtro de **Sobel**, tem uma mascara fixa logo não é necessario especificar a dimesão da mascara.

**Emboss**

2 2\
emboss\
-1 0\
0 1 

**Soma**

3 3\
soma

**Box-to-Box**

3 3\
box\

**Sobel**

3 3\
sobel

---
## Como usar

Para rodar cada um dos arquivos é necessario de duas bibliotecas, **Numpy** e **Pillow**, para aplicar para outras fotos basta mudar a variavel **image_file** colocando o caminho da imagem. Por fim para rodar qualquer um dos arquivo basta rodar o comando:
`python nome_do_arquivo.py`