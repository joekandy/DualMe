CUSTOM_CSS = """
#logo {
    max-width: 200px;
    margin: 0 auto;
    display: block;
}

#person_input, #clothing_input {
    border: 2px solid #051860;
    border-radius: 8px;
    padding: 10px;
}

#model_type {
    background-color: #eccbc8;
    border-radius: 8px;
    padding: 10px;
}

#process_btn {
    background-color: #051860 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
}

#process_btn:hover {
    background-color: #0a2b8c !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(5, 24, 96, 0.2) !important;
}

#output_image {
    border: 2px solid #eccbc8;
    border-radius: 8px;
    padding: 10px;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

h1 {
    color: #051860 !important;
    text-align: center !important;
    margin: 20px 0 !important;
}

.markdown {
    text-align: center !important;
    color: #051860 !important;
}

footer {
    text-align: center !important;
    color: #051860 !important;
    margin-top: 40px !important;
    padding: 20px !important;
    border-top: 1px solid #eccbc8 !important;
}
""" 