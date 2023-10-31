# se a ponta do dedos estiver abaixo da base do dedo
# então dedo está abaixado, senão está acima
# o dedo polegar não tem como fazer esta comparação,
# pois se ele estiver abaixo, provavelmente o dedo está quebrado (kkkkkk)
def verificar_altura_DEDOS(pontos):
    ponta_V = 0
    base_V = 0

    # invertendo o vetor para facilitar o entendimento
    # o indice 0 (zero) será a ponta do dedo, onde normamelte a posição do ponto é mais alta
    for indx, p in enumerate(reversed(pontos)):
        # print(indx, p)
        if indx == 0:
            ponta_V = p[1]
        elif indx == 3:
            base_V = p[1]

    # quanto menor a posição da altura, mais alto o ponto está
    # quanto maior a posição da altura, mais baixo o ponto está
    # COMPARANDO NA VERITICAL
    if ponta_V < base_V:
        return 'acima'
    else:
        return 'abaixo'