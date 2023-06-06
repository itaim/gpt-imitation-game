import logging
import random
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from agents import Agent
from augmentation_service import AugmentationAgent
from index_service import IndexService
from prompts import AGENTS
from storage_service import GCPClient

executor = ThreadPoolExecutor(max_workers=8)
storage = GCPClient()
index_service = IndexService(storage=storage)
augmentation_agent = AugmentationAgent(index_service=index_service, executor=executor)


# session_manager = SessionManager(context_manager)
def shutdown():
    index_service.upload_indices()
    storage.upload_logs()
    logging.info('Orderly shutdown succeeded')


def build_app() -> gr.Blocks:
    block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

    chat_agent = Agent(index_service, augmentation_agent)
    # 	Liam
    username_field = gr.Textbox(
        label='Name',
        placeholder=random.choice(['Olivia','Liam','Emma','Noah']),
        lines=1
    )
    enter_password = gr.Textbox(
        label='Use a password to load an earlier conversation',
        placeholder='******',
        lines=1
    )
    with block:
        # AGENTS.keys()
        agent_names = list(AGENTS)
        # agent_name = 'Joe Rogan'
        # print(f'initial selected agent {agent_names[0]}')
        for agent_name in agent_names:
            with gr.Tab(agent_name):
                print(f'Adding {agent_name} tab')
                # error_box = gr.Textbox(label="Error", visible=False)
                chat_area = gr.Chatbot(elem_id='chatbot', label='chat').style(height=380, container=True)
                chat_state = gr.State()

                with gr.Row():
                    message_field = gr.Textbox(
                        show_label=False,
                        placeholder="what's on your mind?",
                        lines=1,
                    ).style(container=True)
                agent_state = gr.State(agent_name)
                chat_inputs = [message_field, chat_state, username_field, agent_state, enter_password]
                chat_outputs = [chat_area, chat_state, enter_password, message_field]

                def submit_msg(msg, chat_hst, uname, agent_name, pwd):
                    # if not msg:
                    #     return {error_box: gr.update(value="Type a message", visible=True)}
                    # if not uname:
                    #     return {error_box: gr.update(value="Enter a name", visible=True)}
                    # error_box.update(value='')
                    return chat_agent.__call__(msg, chat_hst, uname, agent_name, pwd)

                message_field.submit(chat_agent, inputs=chat_inputs, outputs=chat_outputs)

        with gr.Row():
            with gr.Column(scale=1):
                username_field.render()
            with gr.Column(scale=2):
                enter_password.render()

        return block
