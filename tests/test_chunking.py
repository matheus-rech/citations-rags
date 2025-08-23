import unittest
from src.rag_pdf.chunking import merge_and_chunk, clean_content

class TestChunking(unittest.TestCase):

    def test_merge_and_chunk_basic(self):
        docs = [
            {
                "filename": "doc1.pdf",
                "text": "Page 1 Title\nPage 1 content\fPage 2 Title\nPage 2 content",
                "pages_description": [
                    "Page 1 Title\nPage 1 description",
                    "Page 2 Title\nPage 2 description"
                ]
            }
        ]
        result = merge_and_chunk(docs, remove_first_page=False)
        self.assertEqual(len(result), 2)
        self.assertIn("Page 1 content", result[0]['content'])
        self.assertIn("Page 1 description", result[0]['content'])
        self.assertEqual(result[0]['filename'], 'doc1.pdf')
        self.assertIn("Page 2 content", result[1]['content'])
        self.assertIn("Page 2 description", result[1]['content'])
        self.assertEqual(result[1]['filename'], 'doc1.pdf')

    def test_merge_and_chunk_remove_first_page(self):
        docs = [
            {
                "filename": "doc1.pdf",
                "text": "Page 1 Title\nPage 1 content\fPage 2 Title\nPage 2 content",
                "pages_description": [
                    "Page 1 Title\nPage 1 description",
                    "Page 2 Title\nPage 2 description"
                ]
            }
        ]
        result = merge_and_chunk(docs, remove_first_page=True)
        self.assertEqual(len(result), 1)
        self.assertNotIn("Page 1", result[0]['content'])
        self.assertIn("Page 2 content", result[0]['content'])
        self.assertIn("Page 2 description", result[0]['content'])
        self.assertEqual(result[0]['filename'], 'doc1.pdf')

    def test_merge_and_chunk_unmatched_description(self):
        docs = [
            {
                "filename": "doc1.pdf",
                "text": "Page 1 Title\nPage 1 content",
                "pages_description": [
                    "Unmatched Description"
                ]
            }
        ]
        result = merge_and_chunk(docs, remove_first_page=False)
        self.assertEqual(len(result), 2)
        self.assertIn("Page 1 content", result[0]['content'])
        self.assertEqual(result[0]['filename'], 'doc1.pdf')
        self.assertIn("Unmatched Description", result[1]['content'])
        self.assertEqual(result[1]['filename'], 'doc1.pdf')

    def test_merge_and_chunk_empty_input(self):
        docs = []
        result = merge_and_chunk(docs)
        self.assertEqual(len(result), 0)

    def test_clean_content_whitespace(self):
        pieces = [{"content": "  hello \n world  \n\n extra space  ", "filename": "doc1.pdf"}]
        result = clean_content(pieces)
        self.assertEqual(result[0]['content'], "hello\nworld\nextra space")
        self.assertEqual(result[0]['filename'], "doc1.pdf")

    def test_clean_content_page_numbers(self):
        pieces = [{"content": "some text\n1\nanother line\n12", "filename": "doc1.pdf"}]
        result = clean_content(pieces)
        self.assertEqual(result[0]['content'], "some text\nanother line")
        self.assertEqual(result[0]['filename'], "doc1.pdf")

    def test_clean_content_slide_phrases(self):
        pieces = [
            {"content": "this slide shows data", "filename": "doc1.pdf"},
            {"content": "The Slide explains a concept", "filename": "doc2.pdf"}
        ]
        result = clean_content(pieces)
        self.assertEqual(result[0]['content'], "data")
        self.assertEqual(result[0]['filename'], "doc1.pdf")
        self.assertEqual(result[1]['content'], "a concept")
        self.assertEqual(result[1]['filename'], "doc2.pdf")

if __name__ == '__main__':
    unittest.main()
