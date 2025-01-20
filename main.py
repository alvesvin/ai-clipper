from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    load_indices_from_storage,
    PromptTemplate,
)
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator
)
from llama_index.llms.openai import OpenAI
from moviepy import VideoFileClip
from urllib.parse import urlparse, parse_qs
import whisper
import sys
import yt_dlp
import re
import os
import json
from datetime import datetime, timedelta

llm = OpenAI(temperature=0.1, model_name="gpt-40-mini",
             api_key=os.environ["OPENAI_API_KEY"])

# clip = VideoFileClip("video.mp4")
# # clip.audio.write_audiofile("audio.wav", codec="pcm_s32le")
#
#
# segment = transcript['segments'][0]
#
# clip.subclipped(segment['start'], segment['end']).write_videofile("output.mp4")

# Define a function to calculate duration in seconds

storage_context = StorageContext.from_defaults(persist_dir="./storage")


def is_valid_segment(segment):
    start_time = datetime.strptime(segment['start'], "%H:%M:%S.%f")
    end_time = datetime.strptime(segment['end'], "%H:%M:%S.%f")
    duration = (end_time - start_time).total_seconds()
    return duration >= 1  # Keep segments with duration >= 1 second

# Adjust the timestamps


def adjust_segment(segment):
    # Parse the timestamps
    start_time = datetime.strptime(segment['start'], "%H:%M:%S.%f")
    end_time = datetime.strptime(segment['end'], "%H:%M:%S.%f")

    # Adjust the times
    adjusted_start = (start_time - timedelta(seconds=6)
                      ).strftime("%H:%M:%S.%f")[:-3]
    adjusted_end = (end_time + timedelta(seconds=6)
                    ).strftime("%H:%M:%S.%f")[:-3]

    segment['start'] = adjusted_start
    segment['end'] = adjusted_end

    # Return the adjusted segment
    return segment


def seconds_to_time_format(seconds):
    # Create a timedelta object
    delta = timedelta(seconds=seconds)
    # Convert to a time format using a zero date reference
    time_str = (datetime.min + delta).strftime("%H:%M:%S.%f")
    return time_str[:-3]  # Remove extra microseconds for millisecond precision


def store(url):
    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a],bestaudio[ext=m4a]',
        'outtmpl': {
            'default': 'media/%(id)s/%(id)s.%(ext)s',
            'subtitle': 'media/%(id)s/subs.%(ext)s'
        },
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['pt'],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(url)
        infojson = ydl.extract_info(url, download=False)

    video_id = infojson['id']

    with open('media/%s/info.json' % video_id, 'w') as f:
        json.dump(infojson, f)

    if not os.path.exists(f"media/{video_id}/subs.pt.vtt"):
        model = whisper.load_model("small")
        transcript = model.transcribe(f"media/{video_id}/{video_id}.m4a")

        with open(f"media/{video_id}/subs.pt.vtt", 'w') as f:
            f.write("WEBVTT\nKind: captions\nLanguge: pt\n\n")
            for segment in transcript['segments']:
                f.write(
                    f"{seconds_to_time_format(segment['start'])} --> {seconds_to_time_format(segment['end'])}\n")
                f.write(segment['text'].strip() + "\n\n\n")

    with open('media/%s/subs.pt.vtt' % video_id, 'r') as f:
        lines = f.readlines()

    current_start = None
    current_end = None
    current_content = []
    segments = []

    for line in lines:
        line = line.strip()  # Remove trailing spaces and line breaks

        # Match timecodes (e.g., 00:00:01.000 --> 00:00:05.000)
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})", line)
        if time_match and current_start and current_end:
            # If there is already a segment, save it
            content = " ".join(current_content).strip()
            if len(content) > 1:
                segments.append({
                    "start": current_start,
                    "end": current_end,
                    "content": content
                })
            # Extract start and end times
            current_start = time_match.group(1)
            current_end = time_match.group(2)
            current_content = []

        elif time_match and not current_start and not current_end:
            # Extract start and end times
            current_start = time_match.group(1)
            current_end = time_match.group(2)
        elif current_start and current_end:
            line = re.sub(r"<.*?>", "", line).strip()
            line = re.sub(r"\[.*?\]", "", line).strip()
            if len(line) > 1:
                current_content.append(line)

    # Append the last segment
    content = " ".join(current_content).strip()
    if current_start and current_end and len(content) > 1:
        segments.append({
            "start": current_start,
            "end": current_end,
            "content": content
        })

    segments = [segment for segment in segments if is_valid_segment(segment)]
    segments = [adjust_segment(segment) for segment in segments]

    nodes = []

    for index, segment in enumerate(segments):
        metadata = {
            "start": segment["start"],
            "end": segment["end"],
            "url": url,
            "video_id": video_id,
            "name": infojson["uploader"]
        }

        node = TextNode(
            id_=f"{video_id}-{index}",
            text=segment['content'],
            text_template="{metadata_str}\ncontent: {content}",
            metadata=metadata
        )

        prev_node = nodes[index - 1] if index > 0 else None

        if prev_node:
            node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=f"{video_id}-{index - 1}",
            )
            prev_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=f"{video_id}-{index}",
            )

        nodes.append(node)

    try:
        index = load_index_from_storage(
            storage_context, index_id="videos")
    except Exception:
        index = VectorStoreIndex([], storage_context=storage_context)
        index.set_index_id("videos")

    index.insert_nodes(nodes)
    index.storage_context.persist("./storage")

    print(f"Indexing {len(nodes)} segments")


def search(subject, text):
    index = load_index_from_storage(
        storage_context, index_id="videos")

    print(f"Searching for {subject}")

    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="name",
                operator=FilterOperator.TEXT_MATCH,
                value=subject
            )
        ]
    )

    retriever = index.as_retriever(filters=filters)
    nodes = retriever.retrieve(text)

    # for node in nodes:
    #     print(node.node.get_content(metadata_mode="all"))
    #
    # for index, node in enumerate(nodes):
    #     content = node.node.get_content(metadata_mode="all")
    #     data = dict(line.split(": ", 1) for line in content.splitlines())
    #     clip = VideoFileClip(
    #         f"media/{data['video_id']}/{data['video_id']}.webm")
    #     clip.subclipped(data['start'], data['end']
    #                     ).write_videofile(f"output_cu_{index}.webm")
    #
    # print(nodes)

    prompt_template = PromptTemplate("""
     As informações de segmentos de vídeos são as seguintes:
     ---------------------------
     {segments}
     ---------------------------
     Conforme as informações de segmentos de vídeos e nenhuma outra informação, responda a pergunta.
     O formato da resposta deve ser um objeto JSON contento um resumo e os segmentos encontrados. Se nenhum segmento for encontrado, retorne um objeto contendo um resumo explicando que nada foi encontrado e a propriedade segments como um array vazio. Sempre inclua o "video_id" nos segmentos.
     Exemplo de resposta:
     --------------------------
     {"summary":"Foram encontrados 1 menção de nintendo switch no Canal do Coca no vídeo 'video_id' no minuto 00:00:00 a 00:00:05.","segments":[{"start":"00:00:00","end":"00:00:05","name":"Canal do Coca","content":"Nintendo Switch","video_id":"1234"}]}:

     {"summary":"Não foram encontradas menções de nintendo switch no Canal do Coca","segments":[]}
     --------------------------
     Pergunta: {question}
     Resposta:
     """)

    prompt = prompt_template.format(
        segments="\n\n\n".join(
            [node.node.get_content(metadata_mode="all") for node in nodes]),
        question=text
    )

    response = llm.complete(prompt)

    data = json.loads(response.text)

    print(data["summary"])

    for index, segment in enumerate(data['segments']):
        data = segment
        clip = VideoFileClip(
            f"media/{segment['video_id']}/{segment['video_id']}.mp4")
        clip.subclipped(segment['start'], segment['end']
                        ).write_videofile(f"output_cu_{index}.mp4")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "store":
        return store(sys.argv[2])
    elif cmd == "search":
        return search(sys.argv[2], sys.argv[3])
    else:
        print("Usage: querier <command> [args...]")
        print("Commands:")
        print("  store <url>    - store a video")
        print("  search <text> - search for a video")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
