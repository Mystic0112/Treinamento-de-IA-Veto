import json
import sys
import io
import os
import base64
import matplotlib
matplotlib.use('Agg') # Backend nao-interativo obrigatorio para execucao headless
import matplotlib.pyplot as plt

def executar_e_preencher_notebook(caminho_nb):
    with open(caminho_nb, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    escopo_global = {
        'plt': plt,
        'matplotlib': matplotlib
    }
    
    # Redefinir show no escopo global para nao bloquear
    def noop_show(*args, **kwargs):
        pass
    plt.show = noop_show
    
    total_codigos = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
    print(f"Executando {total_codigos} celulas de codigo no notebook...")
    
    contador = 1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            codigo = "".join(cell['source'])
            print(f"Executando celula {contador}/{total_codigos}...")
            
            # Limpar figuras anteriores
            plt.close('all')
            
            # Capturar stdout
            buffer_stdout = io.StringIO()
            sys_stdout_antigo = sys.stdout
            sys.stdout = buffer_stdout
            
            cell_outputs = []
            
            try:
                # Executar codigo da celula no escopo global compartilhado
                exec(codigo, escopo_global)
            except Exception as e:
                sys.stdout = sys_stdout_antigo
                print(f"Erro na celula {contador}: {e}")
                cell_outputs.append({
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [str(e)]
                })
            else:
                sys.stdout = sys_stdout_antigo
                stdout_texto = buffer_stdout.getvalue()
                if stdout_texto:
                    cell_outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": [line + "\n" for line in stdout_texto.rstrip().split("\n")]
                    })
                    
                # Verificar se foram geradas figuras no matplotlib
                figuras = [plt.figure(num) for num in plt.get_fignums()]
                for fig in figuras:
                    buf_img = io.BytesIO()
                    fig.savefig(buf_img, format='png', bbox_inches='tight', dpi=120)
                    buf_img.seek(0)
                    img_base64 = base64.b64encode(buf_img.read()).decode('utf-8')
                    cell_outputs.append({
                        "data": {
                            "image/png": img_base64,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                    plt.close(fig)
            
            cell['execution_count'] = contador
            cell['outputs'] = cell_outputs
            contador += 1
            
    with open(caminho_nb, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
        
    print(f"\nNotebook {caminho_nb} executado e salvo com todas as saidas e graficos integrados!")

if __name__ == '__main__':
    executar_e_preencher_notebook('Trabalho_Pratico_AV1_ML.ipynb')
