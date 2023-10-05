def hierachic_team(data: list[str]) -> None:
    """
    Обновляет информацию о командах внутри отделов.

    :param data: Список данных о сотруднике, где индекс 1 содержит отдел,
    а индекс 2 - команду.
    :type data: list[str]
    """
    global dep_team_dict

    depart = data[1]
    team = data[2]
    if depart not in dep_team_dict:
        dep_team_dict[depart] = set([team])
    else:
        if team not in dep_team_dict[depart]:
            dep_team_dict[depart].add(team)


def departament_report(data: list[str]) -> None:
    """
    Обновляет статистику по отделам на основе данных о зарплате.

    :param data: Список данных о сотруднике, где индекс 1 содержит отдел,
    а индекс 5 - зарплату.
    :type data: list[str]
    """
    global stats_depart

    depart = data[1]
    salary = int(data[5])

    if depart not in stats_depart:
        stats_depart[depart] = {'Численность': 1,
                                'Вилка зарплат': [salary, salary],
                                'Средняя зарплата': salary}
    else:
        count = stats_depart[depart]['Численность']
        fork_salary = stats_depart[depart]['Вилка зарплат']
        mean_salary = stats_depart[depart]['Средняя зарплата']

        stats_depart[depart]['Численность'] += 1
        stats_depart[depart]['Вилка зарплат'] = [min(fork_salary[0], salary),
                                                 max([fork_salary[1], salary])]

        stats_depart[depart]['Средняя зарплата'] = (mean_salary*(count - 1)
                                                    + salary) / count


def save_data_to_csv(filename: str = r'report.csv') -> None:
    """
    Сохраняет данные в формате CSV.

    :param filename: Имя файла, в который следует сохранить данные.
    :type filename: str
    """
    global stats_depart

    with open(filename, 'w', encoding='UTF-8') as file:

        file.write('Отдел;Численность;Минимальная зарплата;\
                   Максимальная зарплата;Средняя зарплата\n')

        for department, values in stats_depart.items():
            row = '{};{};{};{:.2f}\n'.format(
                department,
                values["Численность"],
                '-'.join(map(str, values["Вилка зарплат"])),
                values["Средняя зарплата"]
            )
            file.write(row)

    print(f'Данные успешно сохранены в {filename}')


def pretify(data: dict) -> str:
    """
    Преобразует словарь в отформатированную строку.

    :param data: Словарь для форматирования.
    :type data: dict
    :return: Отформатированная строка.
    :rtype: str
    """
    formatted_str = ""
    for key, value in data.items():
        formatted_str += f"{key}: {value}\n"

    return formatted_str


def menu() -> None:
    """
    Простое текстовое меню с возможностью выбора опций.

    :return: None
    """
    command_dict = {
        '1': lambda: print(pretify(dep_team_dict)),
        '2': lambda: print(pretify(stats_depart)),
        '3': save_data_to_csv,
    }

    while True:
        print('[1] Иерархия команд')
        print('[2] Сводный отчёт по департаментам')
        print('[3] Сохранить отчёт в формате .csv')
        print('[0] Выход')

        select_button = input('Выберите необходимый пункт меню (0-3): ')

        selected_function = command_dict.get(select_button)

        if selected_function:
            selected_function()
        elif select_button == '0':
            print('Выход из программы.')
            break
        else:
            print('Некорректный выбор. Пожалуйста, выберите пункт меню снова.')


if __name__ == '__main__':
    CSV_FILE_PATH = r'Corp_Summary.csv'

    with open(CSV_FILE_PATH, mode='r+', encoding='UTF-8') as csv:
        dep_team_dict = {}
        stats_depart = {}

        for row in map(lambda x: x[:-1].split(';'), csv.readlines()[1:]):
            hierachic_team(row)
            departament_report(row)

    menu()
