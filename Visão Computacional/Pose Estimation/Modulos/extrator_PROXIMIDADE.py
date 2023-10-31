from Modulos import extrator_ALTURA as altura

# VERIFICAR PLANILHA COM DEMONSTRAÇÃO DOS CÁLCULOS
def verificar_proximidade_DEDOS(pontos):
    pH0 = 0
    pH1 = 0
    pH2 = 0
    pH3 = 0
    pH4 = 0

    for indx, p in enumerate(pontos):
        # 4,8,12,15,20
        # print(indx, p)
        if indx == 4:
            # ponto do polegar
            pH0 = p[0]
            pV0 = p[1]
            # print(pH0)
        if indx == 8:
            # ponta do indicador
            pH1 = p[0]
            pV1 = p[1]
            # print(pH1)
        if indx == 12:
            # ponto do indicador
            pH2 = p[0]
            pV2 = p[1]
            # print(pH2)
        if indx == 15:
            # ponta do medio
            pH3 = p[0]
            pV3 = p[1]
            # print(pH3)
        if indx == 20:
            pH4 = p[0]
            pV4 = p[1]
            # print(pH4)

    if (altura.verificar_altura_DEDOS(pontos[1:5]) and (altura.verificar_altura_DEDOS(pontos[5:9]))) == 'acima':
        if (pH0 - pH1) >= 90:
            polegar = 'afastado'
        elif (pH0 - pH1) < 90:
            polegar = 'proximo'

    if (pH1 - pH2) >= 90:
        indicador = 'afastado'
    elif (pH1 - pH2) < 90:
        indicador = 'proximo'

    if (pH2 - pH3) >= 90:
        medio = 'afastado'
    elif (pH2 - pH3) < 90:
        medio = 'proximo'

    if (pH3 - pH4) >= 90:
        anelar = 'afastado'
    elif (pH3 - pH4) < 90:
        anelar = 'proximo'

    if (pH4 - pH3) >= 90:
        minimo = 'afastado'
    else:
        minimo = 'proximo'

    return polegar
