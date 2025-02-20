class AnalysisUtils:

    @staticmethod
    def get_full_doc_id_used(asin, doc_id_orig):
        doc_id = AnalysisUtils.get_doc_id_used(doc_id_orig)
        return f'{asin}_{doc_id}'

    @staticmethod
    def get_doc_id_used(doc_id_orig):
        if doc_id_orig.startswith('summary') and any([s in doc_id_orig[8:] for s in ['pros', 'cons', 'verdict']]):
            doc_id = 'summary'  # aggregate the verdict, pros and cons in AmaSum
        else:
            doc_id = doc_id_orig
        return doc_id
