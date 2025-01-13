import json

def combine_cascade(dataset, threshold):
    """
    Combine the annotations into a cascade of implications.
    """
    implications = find_logical_implications(dataset, threshold=threshold)
    for i, data in enumerate(dataset):
        data['implications'] = implications[i]
    return implications
        
        
def find_logical_implications(dataset):
    implications = []

    for data in dataset:
        annotations = data["annotations"]
        implication = []
        current_chain = []
        for i in range(len(annotations) - 1):
            hyp = annotations[i]
            ref = annotations[i + 1]

            if 'pairs' in data and i < len(data['pairs']):
                pair = data['pairs'][i]
                if pair['nli'] != 'NAN' and pair['nli'] is True:
                    if not current_chain:
                        current_chain.append(hyp)
                    if hyp == current_chain[-1]:
                        current_chain.append(ref)
                    else:
                        if not current_chain:
                            current_chain.append(hyp)
                        else:
                            implication.append(current_chain)
                            current_chain = []
                else:
                    if current_chain:
                        implication.append(current_chain)
                        current_chain = []
                    else:
                        implication.append([hyp])
            else:
                if current_chain:
                    implication.append(current_chain)
                    current_chain = []
                else:
                    implication.append([hyp])

        if current_chain:
            implication.append(current_chain)
        else:
            implication.append([annotations[-1]])
        cnt = 0
        for imp in implication:
            cnt += len(imp)
        assert cnt == len(annotations)
        implications.append(implication)

    return implications
