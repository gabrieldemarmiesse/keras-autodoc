from docs.autogen import keras_dir, read_page_data
from docs.autogen import get_class_signature, class_to_source_link
from docs.autogen import code_snippet, process_docstring, collect_class_methods, render_function
from docs.autogen import copy_examples
import os
import shutil


def read_file(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def generate(sources_dir, pages):
    """Generates the markdown files for the documentation.

    # Arguments
        sources_dir: Where to put the markdown files.
    """
    template_dir = os.path.join(str(keras_dir), 'docs', 'templates')

    print('Cleaning up existing sources directory.')
    if os.path.exists(sources_dir):
        shutil.rmtree(sources_dir)

    print('Populating sources directory with templates.')
    shutil.copytree(template_dir, sources_dir)

    readme = read_file(os.path.join(str(keras_dir), 'README.md'))
    index = read_file(os.path.join(template_dir, 'index.md'))
    index = index.replace('{{autogenerated}}', readme[readme.find('##'):])
    with open(os.path.join(sources_dir, 'index.md'), 'w', encoding='utf-8') as f:
        f.write(index)

    for page_data in pages:
        classes = read_page_data(page_data, 'classes')

        blocks = []
        for element in classes:
            if not isinstance(element, (list, tuple)):
                element = (element, [])
            cls = element[0]
            subblocks = []
            signature = get_class_signature(cls)
            subblocks.append('<span style="float:right;">' +
                             class_to_source_link(cls) + '</span>')
            if element[1]:
                subblocks.append('## ' + cls.__name__ + ' class\n')
            else:
                subblocks.append('### ' + cls.__name__ + '\n')
            subblocks.append(code_snippet(signature))
            docstring = cls.__doc__
            if docstring:
                subblocks.append(process_docstring(docstring))
            methods = collect_class_methods(cls, element[1])
            if methods:
                subblocks.append('\n---')
                subblocks.append('## ' + cls.__name__ + ' methods\n')
                subblocks.append('\n---\n'.join(
                    [render_function(method, method=True)
                     for method in methods]))
            blocks.append('\n'.join(subblocks))

        methods = read_page_data(page_data, 'methods')

        for method in methods:
            blocks.append(render_function(method, method=True))

        functions = read_page_data(page_data, 'functions')

        for function in functions:
            blocks.append(render_function(function, method=False))

        if not blocks:
            raise RuntimeError('Found no content for page ' +
                               page_data['page'])

        mkdown = '\n----\n\n'.join(blocks)
        # Save module page.
        # Either insert content into existing page,
        # or create page otherwise.
        page_name = page_data['page']
        path = os.path.join(sources_dir, page_name)
        if os.path.exists(path):
            template = read_file(path)
            if '{{autogenerated}}' not in template:
                raise RuntimeError('Template found for ' + path +
                                   ' but missing {{autogenerated}}'
                                   ' tag.')
            mkdown = template.replace('{{autogenerated}}', mkdown)
            print('...inserting autogenerated content into template:', path)
        else:
            print('...creating new page with autogenerated content:', path)
        subdir = os.path.dirname(path)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(mkdown)

    shutil.copyfile(os.path.join(str(keras_dir), 'CONTRIBUTING.md'),
                    os.path.join(str(sources_dir), 'contributing.md'))
    copy_examples(os.path.join(str(keras_dir), 'examples'),
                  os.path.join(str(sources_dir), 'examples'))
