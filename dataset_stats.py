import json
from datasets import load_dataset

if __name__ == '__main__':

    dataset = load_dataset(f"SALT-NLP/feedback_qesconv")
    train_data = dataset['train']

    count_perfect = 0
    count_imperfect = 0

    count_bad_areas = {}
    count_good_areas = {}

    avg_alternative_length = 0
    avg_goal_length = 0

    nr_session = 0
    nr_utternaces = 0

    for ann in train_data:
        nr_utternaces += 1
        nr_session = max(nr_session, ann['conv_index'])
        feedback = json.loads(ann['text'].split("Response:")[1])

        if feedback['perfect']:
            count_perfect += 1
        else:
            count_imperfect += 1
            for badarea in feedback['badareas']:
                if badarea in count_bad_areas:
                    count_bad_areas[badarea] += 1
                else:
                    count_bad_areas[badarea] = 1

            # split alternative into words
            avg_alternative_length += len(feedback['alternative'].split())
            avg_goal_length += len(feedback['feedback'].split())


        for goodarea in feedback['goodareas']:
            if goodarea in count_good_areas:
                count_good_areas[goodarea] += 1
            else:
                count_good_areas[goodarea] = 1


    avg_goal_length /= count_imperfect
    avg_alternative_length /= count_imperfect


    print(f'Nr of sessions: {nr_session+1}')
    print(f'Nr of utterances: {nr_utternaces}')

    print(f'Nr of perfect: {count_perfect}')
    print(f'Nr of imperfect: {count_imperfect}')

    print(f'Good areas: {count_good_areas}')
    print(f'Bad areas: {count_bad_areas}')

    print(f'Avg alternative length: {avg_alternative_length}')
    print(f'Avg goal length: {avg_goal_length}')

    for area in ['Reflections', 'Questions', 'Suggestions', 'Validation', 'Self-disclosure', 'Empathy', 'Professionalism', 'Structure']:
        print(f'{area} {count_bad_areas[area]} {count_good_areas[area]}')


