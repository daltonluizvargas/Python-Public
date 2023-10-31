from Modulos import extrator_ALTURA as altura
pontos = []
posicoes = []


# Para verificar se os dedos estão dobrados ou esticados.
# Esta função faz a comparação da distancia entre pontos
# e adiciona a lista de posições se o dedo está esticado, acima ou abaixo da base do dedo, próximo ou afastado
#
def verificar_posicao_DEDOS(pontos, dedo):
    p3 = 0
    p2 = 0
    p1 = 0
    p0 = 0
    # invertendo o vetor para facilitar o entendimento
    # o indice 0 (zero) será o ponto da base do dedo
    for indx, p in enumerate(reversed(pontos)):
        # print(indx, p[0])
        if indx == 3:
            p3 = p[0] + p[1]
            # print('p3', p3)
        if indx == 2:
            p2 = p[0] + p[1]
            # print('p2', p2)
        if indx == 1:
            p1 = p[0] + p[1]
            # print('p1', p1)
        if indx == 0:
            p0 = p[0] + p[1]
            # print('p0', p0)

    if (p3 - p2) >= 0 and (p2 - p1) >= 0 and (p1 - p0) >= 0:
        return posicoes.append(dedo + ' esticado ' + altura.verificar_altura_DEDOS(pontos))

    else:
        return posicoes.append(dedo + ' dobrado ' + altura.verificar_altura_DEDOS(pontos))

# //////////////////////////////////FUNÇÕES PARA ANÁLISE DE POSIÇÃO DO CORPO///////////////////////////////////////////
def verificar_posicao_CORPO(pontos):
    posicao1 = 0
    posicao2 = 0

    for indx, p in enumerate(pontos):
        # print(indx, p[0])
        if indx == 0:
            posicao1 = p[0] + p[1]
            print('posicao1', posicao1)
        if indx == 1:
            posicao2 = p[0] + p[1]
            print('posicao2', posicao2)

    if (posicao2 - posicao1) >= 0:
        # print('(p3 - p2) = ',(p3 - p2), 'R1')
        # print('(p2 - p1) = ',(p2 - p1), 'R2')
        # print('(p1 - p0) = ',(p1 - p0), 'R3')
        return 'esticado'

    else:
        # print('(p3 - p2) = ',(p3 - p2), 'R1')
        # print('(p2 - p1) = ',(p2 - p1), 'R2')
        # print('(p1 - p0) = ',(p1 - p0), 'R3')
        return 'dobrado'
