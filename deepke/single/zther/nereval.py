from typing import List, Dict, Any
from sklearn.metrics import f1_score, precision_score, recall_score



def to_entities(text: str, labels: List[str]) -> List[dict]:
    """
    Convert a list of token labels into entities.

    Args:
        text (str): Sentences.
        labels (List[str]): A list of labels.
    """
    spans = []
    i, seq_len = 0, len(labels)
    while i < seq_len:
        if labels[i].startswith("B"):
            label_infos = labels[i].split("-")
            span = {
                "start": i,
                "end": i,
                "label": label_infos[1]
            }
            pointer = i + 1
            if pointer == seq_len:
                span["end"] = pointer - 1
                i = pointer
            while pointer < seq_len:
                if labels[pointer].startswith("O") or labels[pointer].startswith("B"):
                    span["end"] = pointer - 1
                    i = pointer
                    break
                elif pointer == seq_len - 1:
                    span["end"] = pointer
                    i = pointer + 1
                    break
                else:
                    pointer += 1
            span["text"] = text[span["start"]: span["end"] + 1]
            spans.append(span)
        else:
            i += 1

    return spans


def evalner(batch_gold_labels: List[List[str]], batch_pred_labels: List[List[str]]) -> Dict[str, Any]:
    """
    Calculate NER metrics: F1, Precision, Recall.

    Args:
        batch_gold_labels (List[List[str]]): Batch gold labels with shape (batch_size, seq_len).
        batch_pred_labels (List[List[str]]): Batch predicted labels with shape (batch_size, seq_len).
    """
    # nsp: not successfully predicted
    scores = {"nsp": []}

    # entity level
    all_gold_entities, all_pred_entities, all_correct_entities = [], [], []
    all_entities_spans = []
    for i, _ in enumerate(batch_gold_labels):
        gold_entities = [
            "[{0}]({1},{2})".format(span["label"], span["start"], span["end"])
            for span in to_entities("O"*len(batch_gold_labels[i]), batch_gold_labels[i])
        ]
        pred_entities = [
            "[{0}]({1},{2})".format(span["label"], span["start"], span["end"])
            for span in to_entities("O"*len(batch_pred_labels[i]), batch_pred_labels[i])
        ]

        entities_spans = [
            (span['start'], span['end'])
            for span in to_entities("O" * len(batch_gold_labels[i]), batch_gold_labels[i])
        ]
        all_entities_spans.append(entities_spans)

        correct_set = set(gold_entities).intersection(set(pred_entities))
        scores["nsp"].append(list(set(gold_entities) - correct_set))
        all_correct_entities.extend(list(correct_set))
        all_gold_entities.extend(gold_entities)
        all_pred_entities.extend(pred_entities)
    scores["entity"] = {
        "p": len(all_correct_entities) / len(all_pred_entities) if len(all_pred_entities) > 0 else 0,
        "r": len(all_correct_entities) / len(all_gold_entities) if len(all_gold_entities) > 0 else 0
    }
    deno = scores["entity"]["p"] + scores["entity"]["r"]
    scores["entity"]["f1"] = 2*scores["entity"]["p"]*scores["entity"]["r"] / deno if deno > 0 else 0

    # token level
    gold_labels, pred_labels = [], []
    for i, _ in enumerate(batch_gold_labels):
        gold_labels.extend(batch_gold_labels[i])
        pred_labels.extend(batch_pred_labels[i])
    scores["token"] = {
        "p": precision_score(gold_labels, pred_labels, average='micro'),
        "r": recall_score(gold_labels, pred_labels, average='micro'),
        "f1": f1_score(gold_labels, pred_labels, average='micro')
    }


    # entity_relax_level
    good_entity_token_labels, pred_entity_token_labels = [], []
    for i, e_span in enumerate(all_entities_spans):
        good_entity_token_labels.extend(batch_gold_labels[i][j[0]:j[1]+1] for j in e_span)
        pred_entity_token_labels.extend(batch_pred_labels[i][j[0]:j[1]+1] for j in e_span)
    good_entity_token_labels = [x for y in good_entity_token_labels for x in y]
    pred_entity_token_labels= [x for y in pred_entity_token_labels for x in y]
    scores["entity_relax"] = {
        "p": precision_score(good_entity_token_labels, pred_entity_token_labels, average='micro'),
        "r": recall_score(good_entity_token_labels, pred_entity_token_labels, average='micro'),
        "f1": f1_score(good_entity_token_labels, pred_entity_token_labels, average='micro')
    }

    return scores


if __name__ =='__main__':


    e = evalner(
        [['B-PER', 'I-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O'], ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC', 'I-LOC']],
        [['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC', 'I-LOC'], ['B-PER', 'I-PER', 'I-PER', 'O', 'O', 'B-LOC', 'O']]
    )

    # ee = to_entities(
    #     '李晓明打球',
    #     ['B-PER', 'I-PER', 'I-PER', 'O', 'O']
    # )
    print(e)
