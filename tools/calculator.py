import operator


def merge_numbers(list_of_numbers):
    return ''.join(list_of_numbers)


def use_operators(op1, oper, op2):

    ops = {
        '+': operator.add,
        '-': operator.sub,
        'times': operator.mul,
        'div': operator.truediv
    }
    op1, op2 = int(op1), int(op2)
    return ops[oper](op1, op2)


def add_negative_numbers(expression):
    for j in range(len(expression)):
        if expression[j] == '-':
            if expression[j+1] == '(':
                continue
            else:
                expression[j+1] = merge_numbers([expression[j], expression[j + 1]])
                expression[j] = '+'
    return expression


def calculate(expression):
    # check the symbols in the expression and calculate until the len of expression is 1
    high_order_operators = []
    low_order_operators = []

    for i in range(len(expression)):
        # assume there is a digit before and after the expression
        if expression[i] == 'div' or expression[i] == 'times':
            high_order_operators.append(i)
        if expression[i] == '+' or expression[i] == '-':
            low_order_operators.append(i)

    new_list_of_expressions_high = expression
    high = False

    if not len(high_order_operators) and not len(low_order_operators):
        result = expression
        return result
    else:
        if len(high_order_operators):
            high = True
            for j in high_order_operators:
                list_to_delete_high = []
                new_result = str(use_operators(expression[j-1], expression[j], expression[j+1]))
                new_list_of_expressions_high[j] = new_result
                list_to_delete_high.append(j-1)
                list_to_delete_high.append(j+1)
                for item in reversed(list_to_delete_high):
                    del new_list_of_expressions_high[item]

        if len(low_order_operators):
            if high:
                low_order_operators = []
                expression = new_list_of_expressions_high
                new_list_of_expressions_low = new_list_of_expressions_high
                for i in range(len(expression)):
                    if expression[i] == '+' or expression[i] == '-':
                        low_order_operators.append(i)
                for j in reversed(low_order_operators):
                    list_to_delete_low = []
                    new_result = str(use_operators(expression[j-1], expression[j], expression[j+1]))
                    new_list_of_expressions_low[j] = new_result
                    list_to_delete_low.append(j-1)
                    list_to_delete_low.append(j+1)
                    for item in reversed(list_to_delete_low):
                        del new_list_of_expressions_low[item]
                result = new_list_of_expressions_low
                return result
            else:
                new_list_of_expressions_low = expression
                for j in reversed(low_order_operators):
                    list_to_delete_low = []
                    new_result = str(use_operators(expression[j-1], expression[j], expression[j+1]))
                    new_list_of_expressions_low[j] = new_result
                    list_to_delete_low.append(j-1)
                    list_to_delete_low.append(j+1)
                    for item in reversed(list_to_delete_low):
                        del new_list_of_expressions_low[item]

                result = new_list_of_expressions_low
                return result

        else:
            result = new_list_of_expressions_high
            return result


def create_big_numbers(expression):
    """
    find if elements contain numbers bigger than 9
    :param expression: original expression
    :return: new expression
    """
    new_expression = []
    temp = []
    for i in range(len(expression)):
        if expression[i].isdigit():
            temp.append(expression[i])
            if i+1 == len(expression):
                if len(temp) > 1:
                    new_expression.append(merge_numbers(temp))
                else:
                    new_expression.append(temp[0])
        else:
            # if list is not empty
            if len(temp):
                # needs to be merged first
                if len(temp) > 1:
                    new_expression.append(merge_numbers(temp))
                else:
                    new_expression.append(temp[0])
                temp = []
                new_expression.append(expression[i])
            else:
                new_expression.append(expression[i])
    new_expression = add_negative_numbers(new_expression)
    return new_expression


def find_parentheses(expression):
    """
    find if elements contain parentheses, if they do save their indexes and extract that expression, if not continue.
    if the size of '(' is different from size od ')' print out warning that user should either enter newly written
    expression or take a better picture of the equation.
    break from calculation with an error
    :param expression: original expression
    :return: new expression
    """
    new_expression = []
    index_left = []
    index_right = []
    for i in range(len(expression)):
        if expression[i] == '(':
            index_left.append(i)
        elif expression[i] == ')':
            index_right.append(i)
    # Check if parentheses were found
    if not len(index_left) and not len(index_right):
        new_expression = expression
        if new_expression[0] in ['+', 'times', 'div']:
            del new_expression[0]
        new_expression = calculate(new_expression)
    elif len(index_left) == len(index_right):
        inner_expression = expression[index_left[0]+1:index_right[0]]
        if inner_expression[0] in ['+', 'times', 'div']:
            del inner_expression[0]
            index_left -= 1
            index_right -= 1
        result_inner_expression = calculate(inner_expression)
        new_expression = expression
        new_expression[index_left[0]] = result_inner_expression[0]
        del new_expression[index_left[0]+1:index_right[0]+1]
        if new_expression[index_left[0]-1].isdigit():
            new_expression.insert(index_left[0], 'times')
        if new_expression[0] in ['+', 'times', 'div']:
            del new_expression[0]
        new_expression = calculate(new_expression)

    elif len(index_left) != len(index_right):
        print('Error in the equation, odd number of parentheses, take another picture, '
              'or write the equation again please')
    return new_expression


def crazy_looping_calculus(expression):
    try:
        exp = create_big_numbers(expression)
        exp = find_parentheses(exp)
        return exp
    except:
        exp = []
        return exp
