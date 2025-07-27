from environment import DATASET_ROOT_DIRS

# Few-shot examples
text_inputs_disrel_examples = [
    "Caption: A handful of palm trees down in our complex, snapped like twigs. Soffit damage a bit less colorful. #Irma\nCoherence Relation: Similar\n", 
    "Caption: Mexico earthquake update: At least 138 people have been reported killed\nCoherence Relation: Complementary\n"
]

text_inputs_clue_sl_hard = [
    "Caption: the feet of a young woman as she is standing on some rocks on the beach\nCoherence Relation: Meta\n", 
    "Caption: young confused girl standing in front of a wardrobe\nCoherence Relation: Visible\n", 
    "Caption: hands typing on a laptop with blank screen\nCoherence Relation: Action\n",
    "Caption: person is excited about the city 's economic prospects\nCoherence Relation: Story\n",
    "Caption: the water was never deeper than your chest .\nCoherence Relation: Subjective\n"
]

text_inputs_clue_ml = [
    "Caption: coffee in a paper cup\n{ \"labels\": [\"Visible\"] }\n",
    "Caption: autumn leaf above tourist attraction\n{ \"labels\": [\"Visible\", \"Meta\"] }\n",
    "Caption: refreshing lemonade made with lemon , strawberry and basil .\n{ \"labels\": [\"Story\"] }\n",
    "Caption: a young person with a hoodie thanks with person together\n{ \"labels\": [\"Action\", \"Story\"] }\n",
    "Caption: film character added to a skull .\n{ \"labels\": [\"Subjective\"] }\n",
    "Caption: chef holding a boiling hen sitting in a pot\n{ \"labels\": [\"Visible\", \"Action\", \"Subjective\"] }\n",
]

text_inputs_clue_ml_csv = [
    "Caption: coffee in a paper cup\nVisible\n",
    "Caption: autumn leaf above tourist attraction\nVisible, Meta\n",
    "Caption: refreshing lemonade made with lemon , strawberry and basil .\nStory\n",
    "Caption: a young person with a hoodie thanks with person together\nAction, Story\n",
    "Caption: film character added to a skull .\nSubjective\n",
    "Caption: chef holding a boiling hen sitting in a pot\nVisible, Action, Subjective\n",
]

text_inputs_tweets_sl = [
    "Caption: welcoming a new little angel into our home within the next 1-3 weeks\nCoherence Relation: Insertion\n",
    "Caption: the 2019 gmc sierra has the world 's first carbon fiber bed\nCoherence Relation: Concretization\n",
    "Caption: learn more about the future of robotics at disrupt sf\nCoherence Relation: Projection\n",
    "Caption: eiffel tower shuts down as snow , freezing rain pummel france\nCoherence Relation: Restatement\n",
    "Caption: my happy place in 4 words my farm , my barn .\nCoherence Relation: Extension\n"
]

image_inputs_tweets_sl = [
    DATASET_ROOT_DIRS['tweets_sl']+"images/01G2beX3dN.jpg",
    DATASET_ROOT_DIRS['tweets_sl']+"images/1wVIXGBzrK.jpg",
    DATASET_ROOT_DIRS['tweets_sl']+"images/03QqHeIOae.jpg",
    DATASET_ROOT_DIRS['tweets_sl']+"images/01y5qXbq8u.jpg",
    DATASET_ROOT_DIRS['tweets_sl']+"images/004ugnbPNp.jpg"
]

image_inputs_clue_ml = [
    DATASET_ROOT_DIRS['clue_ml']+"images/2251.jpg",
    DATASET_ROOT_DIRS['clue_ml']+"images/4022.jpg",
    DATASET_ROOT_DIRS['clue_ml']+"images/4129.jpg",
    DATASET_ROOT_DIRS['clue_ml']+"images/620.jpg",
    DATASET_ROOT_DIRS['clue_ml']+"images/541.jpg",
    DATASET_ROOT_DIRS['clue_ml']+"images/492.jpg",
]

image_inputs_clue_sl_hard = [
    DATASET_ROOT_DIRS['clue_sl_hard']+"images/4004.jpg",
    DATASET_ROOT_DIRS['clue_sl_hard']+"images/2.jpg",
    DATASET_ROOT_DIRS['clue_sl_hard']+"images/3010.jpg",
    DATASET_ROOT_DIRS['clue_sl_hard']+"images/4076.jpg",
    DATASET_ROOT_DIRS['clue_sl_hard']+"images/77.jpg",
]

image_inputs_disrel_examples = [
    DATASET_ROOT_DIRS['disrel_sl']+"images/72977.jpg",
    DATASET_ROOT_DIRS['disrel_sl']+"images/115023.jpg"
]

INSTRUCTION_DEFAULT_ML = {
    'gpt4o' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        "Based on provided information, predict the correct Coherence Relations for the next image-text pair. Output them as a JSON value to the key \"labels\" and do not include any other information in your response.\n",
        "\nCoherence Relations: "
    ],
    'llama32' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        "Based on provided information, predict the correct Coherence Relations for the next image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n",
        "\nCoherence Relations: "
    ],
    'qwen2' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        "Based on provided information, predict the correct Coherence Relations for the next image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n",
        "\nCoherence Relations: "
    ],
    'phi35' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        "Based on provided information, predict the correct Coherence Relations for the next image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n",
        "\nCoherence Relations: "
    ],
    'llava_ov' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        "Based on provided information, predict the correct Coherence Relations for the next image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n",
        "\nCoherence Relations: "
    ],
    'llava16_13b' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        'Based on provided information, predict the correct Coherence Relations for the next image-text pair. Give your predicted labels as comma separated values. Use the format "Coherence Relations: <insert-coherence-relations>" for your response. Do not include any other information in your response.\n',
        "\nCoherence Relations: "
    ],
}

INSTRUCTION_DEFAULT_SL_HARD = {
    'gpt4o' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        "We provide a few examples, each with an input image-text pair, and an output of the correct Coherence Relation.\n",
        "Based on provided information, predict the most applicable Coherence Relation for the next image-text pair. Output only one relation (Visible, Action, Meta, Subjective or Story) and do not include any other information in your response.\n",
        "\nCoherence Relation: "
    ],
    'ft_llama32' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
        "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
        "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
        "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
        "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
        ),
        "We provide a few examples, each with an input image-text pair, and an output of the correct Coherence Relation.\n",
        'Based on provided information, predict the most applicable Coherence Relation for the next image-text pair. Output only one word (Visible, Action, Meta, Subjective or Story) and do not include any other information in your response. Use the format "Coherence Relation: <insert-coherence-relation>" for your response.\n',
        "\nCoherence Relation: "
    ]
}

INSTRUCTION_DEFAULT_TWEETS_SL = {
    'gpt4o' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        # "Coherence is a property of text aiding the processes that construct the mental representation of text in accordance with the author's intent.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Insertion: The salient object described in the image is not explicitly mentioned in the text.\n"
        "- Concretization: Both the text and image contain a mention of the main visual entity.\n"
        "- Projection: The main entity mentioned in the text is implicitly related to the visual objects present in the image.\n"
        "- Restatement: The text directly describes the image contents.\n"
        "- Extension: The image expands upon the story or idea in the text, presenting new elements or elaborations, effectively filling in narrative gaps left by the text.\n"
        ),
        "We provide a few examples, each with an input image-text pair, and an output of the correct Coherence Relation.\n",
        "Based on provided information, predict the most applicable Coherence Relation for the next image-text pair. Output only one relation (Insertion, Concretization, Projection, Restatement or Extension) and do not include any other information in your response.\n",
        "\nCoherence Relation: "
    ]
}

INSTRUCTION_DEFAULT_DISREL = {
    'gpt4o' : [ 
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Similar: The image and text provide the same information and share the same focus. There exists significant overlap in information conveyed between modalities.\n"
        "- Complementary: The image and text do not provide the same information or share the same focus but one modality helps understand the other better.\n"
        ),
        "",
        "Based on provided information, predict the most applicable Coherence Relation for the next image-text pair. Output only one relation (Similar, Complementary) and do not include any other information in your response.\n",
        "\nCoherence Relation: "
    ],

    'ft_llama32' : [
        (
        "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
        "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
        "These are the possible coherence relations you can assign to an image-text pair:\n"
        "- Similar: The image and text provide the same information and share the same focus. There exists significant overlap in information conveyed between modalities.\n"
        "- Complementary: The image and text do not provide the same information or share the same focus but one modality helps understand the other better.\n"
        ),
        "",
        'Based on provided information, predict the most applicable Coherence Relation for the next image-text pair. Output only one word (Similar, Complementary) and do not include any other information in your response. Use the format "Coherence Relation: <insert-coherence-relation>" for your response.\n',
        "\nCoherence Relation: "
    ]
}

INSTRUCTION_COT_ML = {
    'gpt4o' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
            "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
            "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
            "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
            "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
            ),
            "Before assigning coherence relations, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "Now, using your analysis, predict the correct Coherence Relations for the image-text pair. Output them as a JSON value to the key \"labels\" and do not include any other information in your response.\n\n",
            "\nCoherence Relation: "
        ]
    ),
    'llama32' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
            "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
            "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
            "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
            "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
            ),
            "Before assigning coherence relations, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "Now, using your analysis, predict the correct Coherence Relations for the image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n\n",
            "\nCoherence Relation: "
        ]
    ),
    'qwen2' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
            "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
            "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
            "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
            "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
            ),
            "Before assigning coherence relations, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "Now, using your analysis, predict the correct Coherence Relations for the image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n\n",
            "\nCoherence Relation: "
        ]
    ),
    'phi35' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
            "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
            "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
            "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
            "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
            ),
            "Before assigning coherence relations, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "Now, using your analysis, predict the correct Coherence Relations for the image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n\n",
            "\nCoherence Relation: "
        ]
    ),
    'llava_ov' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
            "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
            "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
            "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
            "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
            ),
            "Before assigning coherence relations, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "Now, using your analysis, predict the correct Coherence Relations for the image-text pair. Give your predicted labels as comma separated values. Do not include any other information in your response.\n\n",
            "\nCoherence Relation: "
        ]
    ),
    'llava16_13b' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
            "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
            "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
            "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
            "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
            ),
            "Before assigning coherence relations, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            'Now, using your analysis, predict the correct Coherence Relations for the image-text pair. Give your predicted labels as comma separated values. Use the format "Coherence Relations: <insert-coherence-relations>" for your response. Do not include any other information in your response.\n\n',
            "\nCoherence Relation: "
        ]
    ),
}

INSTRUCTION_COT_SL_HARD = {
    'gpt4o' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Visible: The text presents information that is intended to recognizably characterize what is depicted in the image.\n"
            "- Action: The text describes an extended, dynamic process of which the moment captured in the image is a representative snapshot.\n"
            "- Meta: The text allows the reader to draw inferences not just about the scene depicted in the image but about the production and presentation of the image itself.\n"
            "- Subjective: The text provides information about the speaker's reaction to, or evaluation of, what is depicted in the image.\n"
            "- Story: The text provides a free-standing description of the circumstances depicted in the image, analogous to including instructional, explanatory and other background relations.\n"
            ),
            "We provide a few examples, each with an input image-caption pair, and an output of the correct Coherence Label.\n",
            "Before assigning a coherence relation, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "",
            "Now, using your analysis, predict the most applicable Coherence Relation for the image-text pair. Output only one relation (Visible, Action, Meta, Subjective or Story) and do not include any other information in your response.\n",
            "\nCoherence Relation: "
        ]
    ),

    
}

INSTRUCTION_COT_TWEETS_SL = {
    'gpt4o' : (
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Insertion: The salient object described in the image is not explicitly mentioned in the text.\n"
            "- Concretization: Both the text and image contain a mention of the main visual entity.\n"
            "- Projection: The main entity mentioned in the text is implicitly related to the visual objects present in the image.\n"
            "- Restatement: The text directly describes the image contents.\n"
            "- Extension: The image expands upon the story or idea in the text, presenting new elements or elaborations, effectively filling in narrative gaps left by the text.\n"
            ),
            "We provide a few examples, each with an input image-text pair, and an output of the correct Coherence Relation.\n",
            "Before assigning a coherence relation, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "",
            "Now, using your analysis, predict the most applicable Coherence Relation for the image-text pair. Output only one relation (Insertion, Concretization, Projection, Restatement or Extension) and do not include any other information in your response.\n",
            "\nCoherence Relation: "
        ]
    )
}

INSTRUCTION_COT_DISREL = {
    'gpt4o' : ( 
        [
            (
            "You are an expert linguist and your task is to predict the Coherence Relations of a given image-text pair.\n"
            "A coherence relation captures the structural, logical, and purposeful relationships between an image and its text, capturing the author's intent.\n"
            "These are the possible coherence relations you can assign to an image-text pair:\n"
            "- Similar: The image and text provide the same information and share the same focus. There exists significant overlap in information conveyed between modalities.\n"
            "- Complementary: The image and text do not provide the same information or share the same focus but one modality helps understand the other better.\n"
            ),
            "We provide a few examples, each with an input image-caption pair, and an output of the correct Coherence Label.\n",
            "Before assigning a coherence relation, let's think step by step and analyze the image-text pair in depth.\n",
            "\nAnalysis: "
        ],
        [
            "",
            "",
            "Now, using your analysis, predict the most applicable Coherence Relation for the image-text pair. Output only one relation (Similar or Complementary) and do not include any other information in your response.\n",
            "\nCoherence Relation: "
        ]
    )
}

test_metadata_paths = {
    'clue_ml'       : DATASET_ROOT_DIRS['clue_ml']+'test_ml.json',
    'clue_sl_hard'  : DATASET_ROOT_DIRS['clue_sl_hard']+'test_sl.json',
    'tweets_sl'     : DATASET_ROOT_DIRS['tweets_sl']+'test.json',
    'disrel_sl'     : DATASET_ROOT_DIRS['disrel_sl']+'test.json'
}
train_metadata_paths = {
    'clue_ml'       : DATASET_ROOT_DIRS['clue_ml']+'train_ml.json',
    'clue_sl_hard'  : DATASET_ROOT_DIRS['clue_sl_hard']+'train_sl.json',
    'tweets_sl'     : DATASET_ROOT_DIRS['tweets_sl']+'train.json',
    'disrel_sl'     : DATASET_ROOT_DIRS['disrel_sl']+'train.json'
}
instruction_default_mapping = {
    'clue_ml'       : INSTRUCTION_DEFAULT_ML,
    'clue_sl_hard'  : INSTRUCTION_DEFAULT_SL_HARD,
    'tweets_sl'     : INSTRUCTION_DEFAULT_TWEETS_SL,
    'disrel_sl'     : INSTRUCTION_DEFAULT_DISREL
}
instruction_cot_mapping = {
    'clue_ml'       : INSTRUCTION_COT_ML,
    'clue_sl_hard'  : INSTRUCTION_COT_SL_HARD,
    'tweets_sl'     : INSTRUCTION_COT_TWEETS_SL,
    'disrel_sl'     : INSTRUCTION_COT_DISREL
}
text_examples_mapping = {
    'clue_ml'       : (text_inputs_clue_ml, text_inputs_clue_ml_csv),
    'clue_sl_hard'  : text_inputs_clue_sl_hard,
    'tweets_sl'     : text_inputs_tweets_sl,
    'disrel_sl'     : text_inputs_disrel_examples
}
image_examples_mapping = {
    'clue_ml'       : image_inputs_clue_ml,
    'clue_sl_hard'  : image_inputs_clue_sl_hard,
    'tweets_sl'     : image_inputs_tweets_sl,
    'disrel_sl'     : image_inputs_disrel_examples
}
DATASET_LABELS = {
    'clue_ml' : {
        'Visible': 0, 'Subjective': 1, 'Action': 2, 'Story': 3, 'Meta': 4
    },
    'clue_sl_hard' : {
        'Visible': 0, 'Subjective': 1, 'Action': 2, 'Story': 3, 'Meta': 4
    },
    'disrel_sl' : {
        'Similar': 0, 'Complementary': 1
    },
    'tweets_sl' : {
        'Insertion': 0, 'Concretization': 1, 'Projection': 2, 'Restatement': 3, 'Extension': 4
    }
}