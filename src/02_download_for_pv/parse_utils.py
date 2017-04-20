import xml.etree.ElementTree as ET
from text import get_sentences


HEADING_TAG = 'heading'
PARAGRAPH_TAG = 'p'
UL_TAG = 'ul'
LI_TAG = 'li'
OL_TAG = 'ol'
DESC_OF_DRAWINGS_TAG = 'description-of-drawings'
MIN_PARAGRAPH_LENGTH = 50


def merge_with_previous(curr_node_tag, previous_node_tag, previous_node_text):
    if curr_node_tag == PARAGRAPH_TAG and previous_node_tag == HEADING_TAG:
        return True
    if previous_node_text and len(previous_node_text) < MIN_PARAGRAPH_LENGTH:
        return True
    return False

def get_paragraphs(root):
    paragraphs = []
    previous_node_text = None
    previous_tag = None
    for child in root:
        node_text = None
        if child.tag != DESC_OF_DRAWINGS_TAG:
            node_text = get_node_text(child)
            if node_text.strip():
                if merge_with_previous(child.tag, previous_tag, previous_node_text) and len(paragraphs) > 0:
                    paragraphs[-1] += ' ' + node_text
                else:
                    paragraphs.append(node_text)
        else:
            node_text = extract_desc_of_drawings_paragraph(child)
            paragraphs.append(node_text)

        previous_tag = child.tag
        previous_node_text = node_text
    return paragraphs

def extract_desc_of_drawings_paragraph(node):
    previous_tag = None
    sentences = []
    for child in node:
        node_text = get_node_text(child)
        if child.tag == PARAGRAPH_TAG and previous_tag == HEADING_TAG:
            sentences[-1] += ' ' + node_text
        else:
            # a paragraph in drawings descriptions is treated as a sentence
            if child.tag == PARAGRAPH_TAG:
                node_text = apply_sentence_end(node_text)
            sentences.append(node_text)
        previous_tag = child.tag

    return ' '.join(sentences)

def apply_sentence_end(text):
    if text and text.strip():
        text = text.strip().strip(';.')
        text += '. '
    return text

def itertext_custom(self):
    tag = self.tag
    if not isinstance(tag, basestring) and tag is not None:
        return
    if self.text:
        if tag == LI_TAG:
            yield apply_sentence_end(self.text)
        else:
            yield self.text.replace('\n',' ')
    for e in self:
        for s in e.itertext_custom():
            yield s
        if e.tail:
            yield e.tail

get_node_text = lambda node: ''.join(node.itertext_custom()).strip()


def conc_paragraphs(parag1, parag2):
    return parag1.strip('.') + '.' + ' ' + parag2

def concatenate_sentences_to_paragraphs(paragraphs):
    """
    for 1 sentence paragraphs, concatenate them to the next or previous paragraph depending on context
    """
    for i in range(len(paragraphs)):
        if i >= len((paragraphs)): break
        parag = paragraphs[i]
        sentences = get_sentences(parag)

        if len(sentences) == 1:
            prev_paragraph = paragraphs[i-1] if i-1 >= 0 else None
            next_paragraph = paragraphs[i+1] if i+1 < len(paragraphs) else None

            if (next_paragraph and len(get_sentences(next_paragraph)) == 1):
                # If a series of 1 sentence length paragraphs exist, conc all of them in one paragraph
                while True:
                    if next_paragraph and len(get_sentences(next_paragraph)) == 1:
                        parag = conc_paragraphs(parag, next_paragraph)
                        paragraphs[i] = parag
                        del paragraphs[i+1]

                        # reinitialize for loop
                        next_paragraph = paragraphs[i+1] if i+1 < len(paragraphs) else None
                    else:
                        break

            # otherwise, just concatenate the 1 sentence paragraph to the previous paragraph
            elif prev_paragraph:
#                 print '============== Found prev eligible paragraph'
                prev_paragraph = conc_paragraphs(prev_paragraph, parag)
                paragraphs[i-1] = prev_paragraph
                del paragraphs[i]

            # if this is the first paragraph, then just concatenate it with the next one
            elif next_paragraph:
                parag = conc_paragraphs(parag, next_paragraph)
                paragraphs[i] = parag
                del paragraphs[i+1]

def get_adjusted_paragraphs(root, conc_sentences=True):
    paragraphs = get_paragraphs(root)
    if conc_sentences:
        concatenate_sentences_to_paragraphs(paragraphs)
    return paragraphs


ET.Element.itertext_custom = itertext_custom


