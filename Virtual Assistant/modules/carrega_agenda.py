import pandas as pd
import datetime

hora_atual = datetime.datetime.now()
#print('Hora atual', hora_atual)
hora_atual, minuto_atual = datetime.datetime.time(hora_atual).hour, datetime.datetime.time(hora_atual).minute
#print('Hora atual', hora_atual)
#print('Minuto atual', minuto_atual)
data_atual = datetime.datetime.date(datetime.datetime.today())
#print('Data atual', data_atual)

planilha_agenda = 'path_to_file/agenda.xlsx'
agenda = pd.read_excel(planilha_agenda)
#print(agenda.head())

descricao, responsavel, hora_agenda = [], [], []
#descricao_aviso, responsavel_aviso, hora_aviso = [], [], []

for index, row in agenda.iterrows():
    data = datetime.datetime.date(row['data'])
    #print(data)
    hora_completa = datetime.datetime.strptime(str(row['hora']), '%H:%M:%S')
    #print(hora_completa)
    hora = datetime.datetime.time(hora_completa).hour
    #print(hora)

    if data_atual == data:
        if hora >= hora_atual: # Se a hora do evento agendado for maior ou igual a hora atual. Se não a hora do evento já passou
            descricao.append(row['descricao']), responsavel.append(row['responsavel']), hora_agenda.append(row['hora'])

#print(descricao)
def carregar_agenda():
    if descricao:
        return descricao, responsavel, hora_agenda
    else:
        return False