from typing import Tuple, List

from autorag.nodes.retrieval.base import HybridRetrieval
from autorag.nodes.util import make_generator_callable_param

DAT_PROMPT = """You are an evaluator assessing the retrieval effectiveness of dense
retrieval ( Cosine Distance ) and BM25 retrieval for finding the
correct answer.
## Task :
Given a question and two top1 search results ( one from dense retrieval ,
one from BM25 retrieval ) , score each retrieval method from **0 to 5**
based on whether the correct answer is likely to appear in top2 , top3 , etc.
### ** Scoring Criteria :**
1. ** Direct hit --> 5 points **
- If the retrieved document directly answers the question , assign **5 points **.
2. ** Good wrong result ( High likelihood correct answer is nearby ) --> 3 -4 points **
- If the top1 result is ** conceptually close ** to the correct answer (e. g.,
 mentions relevant entities , related events , partial answer), it indicates the search method is in the right direction.
- Give **4** if it 's very close , **3** if somewhat close.
3. ** Bad wrong result ( Low likelihood correct answer is nearby ) --> 1 -2 points **
- If the top1 result is ** loosely related but misleading ** ( e.g.,shares keywords but changes context ),
correct answers might not be in top2 , top3.
- Give **2** if there 's a small chance correct answers are nearby, **1** if unlikely.
4. ** Completely off - track --> 0 points **
- If the result is ** totally unrelated ** , it means the retrieval method is failing .
---
### ** Given Data :**
- ** Question :** "{question}"
- ** dense retrieval Top1 Result :** "{vector_reference}"
- ** BM25 retrieval Top1 Result :** "{bm25_reference}"
---
### ** Output Format :**
Return two integers separated by a space :
- ** First number :** dense retrieval score.
- ** Second number :** BM25 retrieval score.
- Example output : 3 4
( Vector : 3 , BM25 : 4)
** Do not output any other text .**
"""


class DAT:
	def __init__(self, generator):
		self.generator = generator

	async def calculate_score(self, question, vector_passage, bm25_passage):
		prompt = DAT_PROMPT.format(
			question=question,
			vector_reference=vector_passage,
			bm25_reference=bm25_passage,
		)
		response = await self.generator.llm.acomplete(prompt)

		response = response.text.strip().split(" ")
		assert len(response) == 2, "The llm response have to have two elements"

		vector_score = int(response[0])
		bm25_score = int(response[1])

		if vector_score == 5 and bm25_score == 5:
			return 0.5
		elif vector_score == 5 and bm25_score != 5:
			return 1.0
		elif vector_score != 5 and bm25_score == 5:
			return 0.0
		elif vector_score == 0 and bm25_score == 0:
			return 0.5
		else:
			return vector_score / (vector_score + bm25_score)


class HybridDAT(HybridRetrieval):
	def __init__(
		self, project_dir: str, target_modules, target_module_params, *args, **kwargs
	):
		super().__init__(project_dir, target_modules, target_module_params)
		# set generator module for query expansion
		generator_class, generator_param = make_generator_callable_param(kwargs)
		self.generator = generator_class(project_dir, **generator_param)

	def _pure(
		self, ids: Tuple, scores: Tuple, queries: List[List[str]], top_k: int, **kwargs
	):
		pass
