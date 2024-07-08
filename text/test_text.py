# 指令中的汉文格式转换部分很容易出错，咱们写一个python程序来处理，并用我给你的例子校验，有一个基础版本但有问题，咱们一步步来看：
# 1. annotations都在我给你的文本最后，你从后向前parse，以[]包裹起来的部分为key，比如[9]或[＊5-12]等，value直接设为类似[9]求＝利【宋】【元】【明】。的整条
# 2. 注释处理完就从正文中删掉这部分
# 2. 输出的文本按你写的，用所有标点来分行，每一行正文中如果有命中key，就把对应的value加在这一行的后面

# 请修改以下代码：

import re

def split_text_at_last_line(text):
    lines = text.strip().split('\n')
    last_line = lines[-1]
    all_but_last_line = '\n'.join(lines[:-1])
    return all_but_last_line, last_line

def extract_annotations(annotation_text):
    # 提取注释，并按key字母序排序
    items = annotation_text.strip().split('[')
    annotation_dict = {}
    for item in items:
        if len(item) > 0:
            if item.strip().startswith('＊'):
                key = '[＊]'
                if not key in annotation_dict:
                    annotation_dict[key] = []
                annotation_dict[key].append('[' + item) 
            else:
                match = re.search(r'.*?\]', item)
                if match:
                    key = '[' + match.group(0)
                    annotation_dict[key] = '[' + item
            
    return annotation_dict


def process_text(input_text):
    main_text, annotation_text = split_text_at_last_line(input_text)
    annotation_dict = extract_annotations(annotation_text)

    # 将文本按照句号、问号、叹号、逗号、冒号、分号等分句
    sentences = re.split(r'([。？！：；，])', main_text)
    processed_text = ''
    buffer = ''
    
    for i in range(len(sentences)):
        if i % 2 == 0:
            buffer += sentences[i]
        else:
            buffer += sentences[i]
            buffer = buffer.strip()
            if buffer:
                for key in annotation_dict:
                    if key in buffer:
                        if key == '[＊]':
                            buffer += annotation_dict[key][0]
                            annotation_dict[key].pop(0)
                        else:
                            buffer += annotation_dict[key]
                processed_text += buffer + '\n'
                buffer = ''
    
    return processed_text

# 测试代码
input_text = '''
《大方等大集經》卷59：「佛在王舍國法清淨處時，自然師子座交絡帳。
佛時坐現三十二相，光影表現十方。諸菩薩皆來謁問佛：「菩薩何因緣？有癡者、有黠者、有慧者、有能飛者、有能坐行三昧禪者、有能徹視者、有不能飛者、有不能坐行禪行三昧得定意不能久者、智慧有厚薄者。同菩薩行，何因緣有薄厚？同有心意識、同眼耳鼻口身。何因緣得行異？」
佛言：「善哉，善哉！十方過去佛、現在佛、諸當來佛，皆說人能計心意識眼耳鼻口身，皆說為同法。」
佛言：「人能校計六情為一切，得十方佛智慧。」
佛告諸菩薩言：「諸菩薩有薄厚。」
諸菩薩問佛：「何等為薄厚？」
[7]佛言：「菩薩厚者，謂菩薩行道隨道行深。菩薩薄者，行道不能悉隨行，謂行有多少隨道少，是為菩薩薄。」
諸菩薩問佛：「何等為菩薩常隨道不失行？」
佛言：「謂菩薩常守心意識令不動，歸滅盡種道栽；謂菩薩能守眼令色不著，歸滅盡種道栽；謂菩薩能守耳令聲不著，歸滅盡種道栽；謂菩薩能守鼻令香不著，歸滅盡種道[裁>栽]；謂菩薩能守口令味不著，歸滅盡種道栽；謂菩薩[8]守身令細[9]滑不著，歸滅盡種道栽；菩薩如是能守六情得好惡不動常守滅盡，是為厚隨道深。」」(CBETA, T13, no. 397, p. 394, b9-c2)
[7]〔佛言菩薩〕－【宋】【元】【明】【宮】【聖】。[8]（能）＋守【宋】【元】【明】【宮】【聖】。[9]滑＝濡【聖】。
'''

expected_output = '''
夫[9]求法者，[9]求＝利【宋】【元】【明】。
無知苦求，
無斷[10]習求，[10]習＝集【元】【明】＊。
無造盡證惟道之求。
所以者何？
法無放逸，
有放逸法，
當知苦[＊]習，[＊10-1]習＝集【元】【明】＊。
當為盡證以惟致道；
斯求法者，
無放逸之求[1]也法。[1]也法＝法也【宋】＊【元】＊【明】＊。
舍利弗！
無有塵、離婬塵，
其染污者，
即為在邊；
斯求法者，
無婬樂之求[＊]也法。[＊1-1]也法＝法也【宋】＊【元】＊【明】＊。
'''

processed_text = process_text(input_text)
print(processed_text)
# assert processed_text.strip() == expected_output.strip(), "输出不匹配预期结果！"

# print("输出匹配预期结果。")

