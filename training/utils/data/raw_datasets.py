import json
import re
from datasets import load_dataset


class PromptRawDataset(object):

    def __init__(self, args):
        self.data = []
        self.no_train_split = args.no_train_split

    def get_unsup_dataset(self):
        data = []
        for split in self.data:
            if self.no_train_split and split == "train":
                continue
            for item in self.data[split]:
                data.append({"prompt": item["prompt"].strip()})
        return data

    def get_sft_dataset(self):
        data = []
        for split in self.data:
            if self.no_train_split and split == "train":
                continue
            for item in self.data[split]:
                data.append({"prompt": item["prompt"].strip(), "chosen": item["chosen"].strip()})
        return data

    def get_rm_dataset(self):
        data = []
        for split in self.data:
            if self.no_train_split and split == "train":
                continue
            for item in self.data[split]:
                data.append({"prompt": item["prompt"].strip(), "chosen": item["chosen"].strip(), "reject": item["rejected"].strip()})
        return data

    def get_prompt(self, sample):
        return sample['prompt'] # + " "

    def get_chosen(self, sample):
        return sample['chosen'] #+ "\n\nHuman:"

    def get_rejected(self, sample):
        return sample['reject'] #+ "\n\nHuman:"

    def get_prompt_and_chosen(self, sample):
        return self.get_prompt(sample) + " " + self.get_chosen(sample)

    def get_prompt_and_rejected(self, sample):
        return self.get_prompt(sample) + " " + self.get_rejected(sample)


class LocalDataset(PromptRawDataset):

    def __init__(self, args, dataset_name_or_path):
        super().__init__(args)
        with open(dataset_name_or_path) as f:
            self.data = [json.loads(line.strip()) for line in f]

    def get_unsup_dataset(self):
        return self.data

    def get_sft_dataset(self):
        return self.data

    def get_rm_dataset(self):
        return list(filter(lambda x: x["reject"], self.data))


class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, args, dataset_name_or_path):
        super().__init__(args)
        self.dataset_name = dataset_name_or_path
        success = False
        while not success:
            try:
                self.data = load_dataset(dataset_name_or_path)
                success = True
            except:
                pass
        self.dataset_name_clean = dataset_name_or_path.replace("/", "_")

# English dataset
class TLDRDataset(PromptRawDataset):

    def __init__(self, args, dataset_name_or_path):
        super().__init__(args)
        self.dataset_name = "CarperAI/openai_summarize_tldr"
        self.dataset_name_clean = "CarperAI_openai_summarize_tldr"
        success = False
        while not success:
            try:
                self.data = load_dataset("CarperAI/openai_summarize_tldr")
                success = True
            except:
                pass

    def get_sft_dataset(self):
        data = []
        for split in self.data:
            if self.no_train_split and split == "train":
                continue
            for item in self.data[split]:
                data.append({"prompt": item["prompt"].strip() + " ", "chosen": item["label"].strip()})
        return data

    def get_rm_dataset(self):
        raise RuntimeError("No rejected for this dataset")

    def get_prompt(self, sample):
        return "Human: Please summarize the following Reddit post from the writer's perspective as brief and accurate as possible.\n\n" + sample['prompt'].replace("TL;DR:", "").strip() + "\n\nAssistant: TL;DR: "

    def get_rejected(self, sample):
        raise RuntimeError("No rejected for this dataset")

# English dataset
class TLDRComparisonDataset(PromptRawDataset):

    def __init__(self, args, dataset_name_or_path):
        super().__init__(args)
        self.dataset_name = "CarperAI/openai_summarize_comparisons"
        self.dataset_name_clean = "CarperAI_openai_summarize_comparisons"
        success = False
        while not success:
            try:
                self.data = load_dataset("CarperAI/openai_summarize_comparisons")
                success = True
            except:
                pass

    def get_prompt(self, sample):
        return "Human: Please summarize the following Reddit post from the writer's perspective as brief and accurate as possible.\n\n" + sample['prompt'] + "\n\nAssistant: TL;DR: "

    def get_chosen(self, sample):
        return sample['chosen'].replace("TL;DR:", "").strip() #+ "\n\nHuman:"

    def get_rejected(self, sample):
        return sample['reject'].replace("TL;DR:", "").strip() #+ "\n\nHuman:"


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):

    def __init__(self, args, dataset_name_or_path):
        super().__init__(args)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"
        success = False
        while not success:
            try:
                self.data = load_dataset("openai/webgpt_comparisons")
                success = True
            except:
                pass

    def get_unsup_dataset(self):
        data = []
        for split in self.data:
            for item in self.data[split]:
                data.append({"prompt": "Human: " + item['question']['full_text'] + "\n\nAssistant: "})
        return data

    def get_sft_dataset(self):
        data = []
        for split in self.data:
            for item in self.data[split]:
                if float(item['score_0']) < float(item['score_1']):
                    chosen = item['answer_1']
                else:
                    chosen = item['answer_0']
                chosen = re.sub(r" [\(\[].*?[\)\]]", "", chosen)
                chosen = re.sub(r"[\(\[].*?[\)\]]", "", chosen)
                data.append({"prompt": "Human: " + item['question']['full_text'] + "\n\nAssistant: ", "chosen": chosen.strip()})
        return data

    def get_rm_dataset(self):
        data = []
        for split in self.data:
            for item in self.data[split]:
                if float(item['score_0']) < float(item['score_1']):
                    chosen, reject = item['answer_1'], item['answer_0']
                else:
                    chosen, reject = item['answer_0'], item['answer_1']
                chosen = re.sub(r" [\(\[].*?[\)\]]", "", chosen)
                chosen = re.sub(r"[\(\[].*?[\)\]]", "", chosen)
                reject = re.sub(r" [\(\[].*?[\)\]]", "", reject)
                reject = re.sub(r"[\(\[].*?[\)\]]", "", reject)
                data.append({"prompt": "Human: " + item['question']['full_text'] + "\n\nAssistant: ", "chosen": chosen.strip(), "reject": reject.strip()})
        return data
