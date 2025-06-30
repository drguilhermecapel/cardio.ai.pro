# Credenciais de Acesso ao CardioAI Pro

Este documento contém as credenciais padrão para acessar o sistema CardioAI Pro.

## Credenciais de Administrador

Para acessar o sistema como administrador, utilize as seguintes credenciais:

- **Email/Usuário**: `admin@cardioai.com`
- **Senha**: `admin123`

## Observações Importantes

1. **Segurança**: Estas são credenciais padrão e devem ser alteradas após o primeiro acesso ao sistema em um ambiente de produção.

2. **Geração de Senha**: Se a senha padrão (`admin123`) for considerada insegura pelo sistema, uma nova senha será gerada automaticamente durante a inicialização do banco de dados. Neste caso, a senha gerada será exibida no console do backend.

3. **Recuperação de Senha**: Se você esquecer a senha de administrador, você pode executar o script `get_admin_password.py` na pasta backend para recuperá-la:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\backend
   python get_admin_password.py
   ```

## Fluxo de Login

1. Acesse o frontend em `http://localhost:5173`
2. Na tela de login, insira o email e senha do administrador
3. Clique no botão "Entrar"

## Solução de Problemas

Se você encontrar problemas ao fazer login:

1. **Verifique se o backend está em execução**: O servidor backend deve estar rodando na porta 8000.

2. **Verifique os logs do backend**: Observe o console onde o backend está sendo executado para verificar se há erros de autenticação.

3. **Verifique as configurações de CORS**: Se houver erros de CORS, verifique se o frontend está configurado para se comunicar corretamente com o backend.

4. **Reinicie o banco de dados**: Em caso de problemas persistentes, você pode reinicializar o banco de dados:

   ```powershell
   cd C:\caminho\para\cardio.ai.pro\backend
   python init_database.py
   ```

## Usuários Adicionais

O sistema permite a criação de usuários adicionais com diferentes níveis de acesso:

1. **Médicos**: Podem acessar e gerenciar pacientes, consultas e análises de ECG.
2. **Técnicos**: Podem realizar e enviar exames de ECG para análise.
3. **Pacientes**: Podem visualizar seus próprios exames e resultados.

Para criar novos usuários, utilize a interface de administração após fazer login como administrador.