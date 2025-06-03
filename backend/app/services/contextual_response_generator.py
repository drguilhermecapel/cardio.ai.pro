"""
Contextual Response Generator for Non-ECG Images
Generates educational and helpful responses based on image classification.
"""

import logging
from typing import Any

from app.models.user import UserSession
from app.services.non_ecg_image_classifier import NonECGImageClassifier

logger = logging.getLogger(__name__)


class ContextualResponseGenerator:
    """Generate contextual, educational responses for non-ECG images."""

    def __init__(self) -> None:
        """Initialize the contextual response generator."""
        from app.services.adaptive_feedback_service import adaptive_feedback_service
        self.adaptive_feedback_service = adaptive_feedback_service
        self.classifier = NonECGImageClassifier()
        self.ecg_examples = self._load_ecg_examples()

    def generate_response(
        self,
        category: str,
        confidence: float,
        user_session: UserSession | None = None
    ) -> dict[str, Any]:
        """Generate contextual response based on image category and user history."""

        base_response = self._get_base_response(category, confidence)

        if user_session:
            base_response = self._personalize_response(base_response, user_session)

        base_response.update({
            'tips': base_response.get('tips', []),
            'ecg_visual_guide': self._generate_ecg_visual_guide(),
            'helpful_actions': self._get_helpful_actions(),
            'privacy_notice': "🔒 Sua imagem não foi armazenada por não ser um ECG.",
            'learn_more_link': "Saiba mais sobre ECGs →"
        })

        return base_response

    def _get_base_response(self, category: str, confidence: float) -> dict[str, Any]:
        """Get base response for specific category."""

        responses = {
            'x_ray': {
                'message': "Detectamos uma imagem de raio-X.",
                'explanation': "Raios-X são exames importantes, mas analisamos especificamente eletrocardiogramas (ECGs).",
                'tips': [
                    "ECGs mostram a atividade elétrica do coração, não estruturas ósseas",
                    "Procure por um documento com ondas cardíacas em papel milimetrado",
                    "ECGs têm múltiplas derivações (I, II, III, aVR, aVL, aVF, V1-V6)"
                ]
            },

            'medical_other': {
                'icon': '🏥',
                'message': "Detectamos que você enviou um exame médico, mas não é um ECG.",
                'explanation': "Este parece ser outro tipo de exame médico. Nosso sistema é especializado em análise de eletrocardiogramas (ECG).",
                'tips': [
                    "Procure por papel milimetrado com ondas cardíacas",
                    "ECGs mostram 12 derivações diferentes",
                    "Verifique se há informações do paciente no documento"
                ],
                'suggestion': "Se você precisa analisar este exame, recomendamos consultar um profissional especializado na área.",
                'helpful_tip': "Um ECG tem um padrão característico de grade milimetrada com ondas cardíacas.",
                'tone': 'professional_helpful'
            },

            'xray': {
                'icon': '🦴',
                'message': "Esta parece ser uma radiografia (raio-X).",
                'explanation': "Radiografias são excelentes para visualizar ossos e estruturas internas, mas nosso sistema analisa especificamente eletrocardiogramas (ECG).",
                'tips': [
                    "ECGs são gráficos de ondas, não imagens anatômicas",
                    "Procure por papel com grade milimetrada",
                    "ECGs mostram ritmo cardíaco ao longo do tempo"
                ],
                'suggestion': "Para análise de radiografias, consulte um radiologista ou médico especializado.",
                'helpful_tip': "ECGs mostram a atividade elétrica do coração, não imagens anatômicas como raios-X.",
                'tone': 'educational'
            },

            'prescription': {
                'icon': '💊',
                'message': "Identificamos uma receita médica.",
                'explanation': "Receitas são importantes para o tratamento, mas nosso foco é na análise de eletrocardiogramas (ECG).",
                'suggestion': "Para dúvidas sobre medicamentos, consulte seu médico ou farmacêutico.",
                'helpful_tip': "ECGs são exames que registram os batimentos cardíacos em papel milimetrado.",
                'tone': 'friendly_professional'
            },

            'lab_results': {
                'icon': '🧪',
                'message': "Estes parecem ser resultados de exames laboratoriais.",
                'explanation': "Exames de sangue e outros testes laboratoriais são fundamentais para o diagnóstico, mas analisamos especificamente ECGs.",
                'suggestion': "Para interpretação de exames laboratoriais, consulte o médico que os solicitou.",
                'helpful_tip': "ECGs são diferentes dos exames de sangue - eles mostram o ritmo e a atividade elétrica do coração.",
                'tone': 'informative'
            },

            'photo_person': {
                'icon': '📸',
                'message': "Esta é uma foto de pessoa.",
                'explanation': "Embora pessoas sejam importantes, nosso sistema analisa documentos médicos específicos: eletrocardiogramas (ECG).",
                'suggestion': "Para análise de ECG, você precisa de um documento médico com o registro dos batimentos cardíacos.",
                'helpful_tip': "ECGs são impressos em papel especial com grade milimetrada e mostram ondas cardíacas.",
                'tone': 'friendly_light',
                'humor': "📷 Bela foto! Mas analisamos batimentos, não sorrisos! 😊"
            },

            'screenshot': {
                'icon': '📱',
                'message': "Esta é uma captura de tela.",
                'explanation': "Capturas de tela podem conter informações úteis, mas precisamos do documento ECG original para análise precisa.",
                'suggestion': "Se você tem um ECG em formato digital, tente fazer upload do arquivo original ou uma foto clara do documento impresso.",
                'helpful_tip': "Para melhor análise, use fotos diretas do ECG em papel ou arquivos digitais originais.",
                'tone': 'helpful_technical'
            },

            'nature': {
                'icon': '🌿',
                'message': "Que bela paisagem!",
                'explanation': "A natureza é relaxante e faz bem ao coração, mas nosso sistema analisa especificamente eletrocardiogramas médicos.",
                'suggestion': "Para análise cardíaca, você precisará de um ECG realizado por um profissional de saúde.",
                'helpful_tip': "ECGs capturam os 'ritmos da natureza' do seu coração em papel milimetrado.",
                'tone': 'friendly_light',
                'humor': "🌱 A natureza é ótima para o coração, mas precisamos ver os batimentos no papel! 💚"
            },

            'object': {
                'icon': '📦',
                'message': "Identificamos um objeto na imagem.",
                'explanation': "Objetos diversos podem ser interessantes, mas nosso sistema é especializado em análise de eletrocardiogramas (ECG).",
                'suggestion': "Para análise cardíaca, você precisará de um documento ECG de um exame médico.",
                'helpful_tip': "ECGs são documentos médicos específicos com ondas que representam batimentos cardíacos.",
                'tone': 'neutral_helpful'
            },

            'text_document': {
                'icon': '📄',
                'message': "Este é um documento de texto.",
                'explanation': "Documentos de texto são úteis para informações, mas analisamos especificamente eletrocardiogramas (ECG) médicos.",
                'suggestion': "Se você tem um relatório médico com ECG, procure pela parte que contém as ondas cardíacas em grade milimetrada.",
                'helpful_tip': "ECGs têm características visuais específicas: grade milimetrada e ondas cardíacas.",
                'tone': 'instructional'
            },

            'other_medical_device': {
                'icon': '⚕️',
                'message': "Detectamos saída de outro dispositivo médico.",
                'explanation': "Existem muitos dispositivos médicos importantes, mas nosso sistema é otimizado especificamente para eletrocardiogramas (ECG).",
                'suggestion': "Se este dispositivo produz ECGs, procure pela saída que mostra ondas cardíacas em papel milimetrado.",
                'helpful_tip': "ECGs têm formato padronizado com 12 derivações e grade milimetrada característica.",
                'tone': 'professional_specific'
            },

            'monitor_screen': {
                'icon': '🖥️',
                'message': "Esta parece ser uma tela de monitor médico.",
                'explanation': "Monitores médicos mostram informações valiosas, mas para análise detalhada precisamos do ECG impresso ou exportado.",
                'suggestion': "Tente obter uma impressão do ECG ou exporte os dados do monitor para análise mais precisa.",
                'helpful_tip': "ECGs impressos oferecem melhor resolução e detalhes para análise computacional.",
                'tone': 'technical_helpful'
            },

            'handwritten': {
                'icon': '✍️',
                'message': "Identificamos anotações manuscritas.",
                'explanation': "Anotações médicas são importantes para o contexto, mas analisamos os traçados eletrônicos do ECG.",
                'suggestion': "Se há um ECG junto com essas anotações, foque na parte com as ondas cardíacas em papel milimetrado.",
                'helpful_tip': "ECGs são traçados eletrônicos precisos, diferentes de anotações manuais.",
                'tone': 'clarifying'
            },

            'food': {
                'icon': '🍕',
                'message': "Hmm... delicioso!",
                'explanation': "Uma boa alimentação é importante para a saúde cardíaca, mas analisamos eletrocardiogramas médicos.",
                'tips': [
                    "ECGs são documentos médicos com ondas cardíacas",
                    "Procure por papel milimetrado com traçados",
                    "ECGs geralmente têm 12 derivações diferentes"
                ],
                'suggestion': "Para análise cardíaca, você precisará de um ECG realizado por um profissional de saúde.",
                'helpful_tip': "ECGs mostram como seu coração 'digere' os impulsos elétricos! 😄",
                'tone': 'friendly_humorous',
                'humor': "🍽️ Analisamos batimentos, não receitas! Mas uma dieta saudável faz bem ao coração! 💪"
            },

            'document': {
                'icon': '📋',
                'message': "Este é um documento geral.",
                'explanation': "Documentos podem conter informações médicas importantes, mas precisamos especificamente de eletrocardiogramas (ECG).",
                'suggestion': "Se este documento contém um ECG, procure pela seção com ondas cardíacas em papel milimetrado.",
                'helpful_tip': "ECGs são facilmente identificáveis pela grade milimetrada e padrões de ondas cardíacas.",
                'tone': 'instructional'
            },

            'unknown': {
                'icon': '🤔',
                'message': "Não conseguimos identificar claramente o conteúdo desta imagem.",
                'explanation': "Para uma análise precisa, nosso sistema precisa de eletrocardiogramas (ECG) claros e bem definidos.",
                'tips': [
                    "Verifique se a imagem está clara e bem iluminada",
                    "ECGs têm grade milimetrada característica",
                    "Procure por ondas cardíacas no documento"
                ],
                'suggestion': "Tente tirar uma nova foto com boa iluminação de um documento ECG, ou verifique se o arquivo não está corrompido.",
                'helpful_tip': "ECGs têm características visuais distintas que facilitam a identificação.",
                'tone': 'helpful_troubleshooting'
            }
        }

        response = responses.get(category, responses['unknown'])
        result: dict[str, Any] = response.copy() if isinstance(response, dict) else {}

        if category == 'food':
            result['humor_response'] = self.generate_humor_response(category)

        return result

    def _personalize_response(
        self,
        base_response: dict[str, Any],
        user_session: UserSession
    ) -> dict[str, Any]:
        """Personalize response based on user session history."""

        # Add adaptive suggestions
        result = dict(base_response)
        result['adaptive_suggestions'] = self.get_adaptive_suggestions(base_response.get('category', 'unknown'), user_session)

        result['learning_stage'] = getattr(user_session, 'learning_stage', 'beginner')

        result['encouragement'] = "Continue tentando! Estamos aqui para ajudar você a analisar ECGs corretamente."

        return result

    def _generate_ecg_visual_guide(self) -> dict[str, Any]:
        """Generate visual guide showing what ECGs look like."""

        return {
            'title': "Como identificar um ECG:",
            'characteristics': [
                {
                    'feature': "Grade milimetrada",
                    'description': "Fundo com pequenos quadrados formando uma grade",
                    'icon': "📐"
                },
                {
                    'feature': "12 derivações",
                    'description': "Geralmente mostra 12 diferentes 'visões' do coração (I, II, III, aVR, aVL, aVF, V1-V6)",
                    'icon': "🔢"
                },
                {
                    'feature': "Ondas cardíacas",
                    'description': "Padrões de ondas que representam os batimentos do coração",
                    'icon': "〰️"
                },
                {
                    'feature': "Informações do paciente",
                    'description': "Nome, idade, data do exame, frequência cardíaca",
                    'icon': "👤"
                },
                {
                    'feature': "Papel especial",
                    'description': "Geralmente em papel térmico ou especial para ECG",
                    'icon': "📄"
                }
            ],
            'example_description': "Um ECG típico parece com um gráfico de ondas em papel milimetrado, mostrando o ritmo cardíaco ao longo do tempo.",
            'common_formats': [
                "Papel impresso (mais comum)",
                "Arquivo PDF de ECG",
                "Imagem digital de alta resolução",
                "Foto clara do documento original"
            ]
        }

    def _get_helpful_actions(self) -> list[dict[str, str]]:
        """Get list of helpful actions user can take."""

        return [
            {
                'action': "Tirar nova foto",
                'description': "Fotografe um ECG real com boa iluminação",
                'icon': "📷"
            },
            {
                'action': "Ver exemplos de ECG",
                'description': "Veja como um ECG deve parecer",
                'icon': "👁️"
            },
            {
                'action': "Tutorial sobre ECGs",
                'description': "Aprenda mais sobre eletrocardiogramas",
                'icon': "🎓"
            },
            {
                'action': "Falar com suporte",
                'description': "Precisa de ajuda? Entre em contato",
                'icon': "💬"
            }
        ]

    def _load_ecg_examples(self) -> list[dict[str, Any]]:
        """Load ECG examples for educational purposes."""


        return [
            {
                'type': "Normal ECG",
                'description': "ECG com ritmo sinusal normal",
                'characteristics': ["Ondas P regulares", "Complexos QRS estreitos", "Frequência 60-100 bpm"]
            },
            {
                'type': "12-lead ECG",
                'description': "ECG padrão com 12 derivações",
                'characteristics': ["12 diferentes visões do coração", "Grade milimetrada", "Informações do paciente"]
            },
            {
                'type': "Rhythm Strip",
                'description': "Tira de ritmo contínua",
                'characteristics': ["Registro longo de uma derivação", "Útil para análise de arritmias"]
            }
        ]

    def generate_educational_content(self, category: str) -> dict[str, Any]:
        """Generate educational content specific to the detected category."""

        educational_content = {
            'medical_other': {
                'title': "Diferenças entre ECG e outros exames médicos",
                'description': "ECGs são documentos médicos especializados que diferem de outros exames",
                'content': [
                    "ECGs medem atividade elétrica do coração",
                    "Outros exames podem medir diferentes aspectos da saúde",
                    "Cada exame tem sua importância específica no diagnóstico"
                ],
                'comparison': "ECG vs. outros exames médicos",
                'key_features': [
                    "Grade milimetrada característica",
                    "Ondas P, QRS e T visíveis",
                    "Múltiplas derivações (I, II, III, aVR, aVL, aVF, V1-V6)",
                    "Informações do paciente e configurações do equipamento"
                ]
            },

            'xray': {
                'title': "ECG vs. Radiografia: Entenda a diferença",
                'description': "Radiografias e ECGs são exames complementares com propósitos diferentes",
                'content': [
                    "Radiografias mostram estruturas anatômicas (ossos, órgãos)",
                    "ECGs mostram atividade elétrica do coração",
                    "Ambos são importantes para diagnóstico cardíaco completo"
                ],
                'comparison': "Imagem anatômica vs. atividade elétrica",
                'key_features': [
                    "Grade milimetrada característica",
                    "Ondas P, QRS e T visíveis",
                    "Múltiplas derivações (I, II, III, aVR, aVL, aVF, V1-V6)",
                    "Informações do paciente e configurações do equipamento"
                ]
            },

            'photo_person': {
                'title': "Por que precisamos de documentos médicos?",
                'description': "Fotos pessoais não fornecem dados médicos necessários para análise",
                'content': [
                    "Fotos de pessoas não contêm dados médicos mensuráveis",
                    "ECGs fornecem dados precisos sobre função cardíaca",
                    "Análise médica requer documentação específica"
                ],
                'comparison': "Foto pessoal vs. documento médico",
                'key_features': [
                    "Grade milimetrada característica",
                    "Ondas P, QRS e T visíveis",
                    "Múltiplas derivações (I, II, III, aVR, aVL, aVF, V1-V6)",
                    "Informações do paciente e configurações do equipamento"
                ]
            }
        }

        return educational_content.get(category, {
            'title': "Sobre eletrocardiogramas (ECG)",
            'description': "ECGs são documentos médicos especializados para análise cardíaca",
            'content': [
                "ECGs registram a atividade elétrica do coração",
                "São fundamentais para diagnóstico de problemas cardíacos",
                "Requerem interpretação por profissionais qualificados"
            ],
            'comparison': "Documento médico especializado",
            'key_features': [
                "Grade milimetrada característica",
                "Ondas P, QRS e T visíveis",
                "Múltiplas derivações (I, II, III, aVR, aVL, aVF, V1-V6)",
                "Informações do paciente e configurações do equipamento"
            ]
        })

    def get_adaptive_suggestions(
        self,
        category: str,
        user_session: Any | None = None
    ) -> list[str]:
        """Get adaptive suggestions based on user's upload history."""

        user_history = []
        if user_session and hasattr(user_session, 'category_history'):
            user_history = user_session.category_history

        suggestions = []

        if user_history and hasattr(user_history, 'count') and user_history.count('photo_person') > 2:
            suggestions.append("Parece que você está enviando fotos pessoais. Para análise cardíaca, precisamos de documentos ECG médicos.")

        medical_categories = ['medical_other', 'xray', 'prescription', 'lab_results']
        if user_history and hasattr(user_history, '__contains__') and any(cat in user_history for cat in medical_categories):
            suggestions.append("Vemos que você tem documentos médicos. Procure especificamente por um ECG (eletrocardiograma) com ondas cardíacas.")

        if user_history and hasattr(user_history, '__contains__') and 'screenshot' in user_history:
            suggestions.append("Para melhor análise, use o arquivo original do ECG ou uma foto direta do documento impresso.")

        if not suggestions:
            suggestions.append("Dica: ECGs são facilmente identificáveis pela grade milimetrada e ondas cardíacas características.")

        return suggestions

    def get_base_response_categories(self) -> dict[str, Any]:
        """Get list of supported response categories."""
        return {
            'categories': [
                'x_ray', 'medical_other', 'prescription', 'lab_results',
                'medical_device', 'monitor_screen', 'document', 'text_document',
                'handwritten', 'photo_person', 'nature', 'object', 'food', 'screenshot'
            ],
            'total_count': 14
        }

    def generate_humor_response(self, category: str) -> str | None:
        """Generate light humor when appropriate."""

        humor_responses = {
            'food': [
                "🍎 Uma maçã por dia mantém o médico longe, mas um ECG nos mantém próximos! 😄",
                "🥗 Comida saudável faz bem ao coração, ECGs nos mostram como ele está! 💚",
                "🍕 Pizza é boa para a alma, ECG é bom para o coração! 😊"
            ],

            'nature': [
                "🌳 A natureza acalma o coração, ECGs nos mostram seu ritmo! 🎵",
                "🌸 Flores são belas, mas as ondas do ECG também têm sua beleza! 💖",
                "🏔️ Montanhas têm picos, ECGs também! Mas os do coração são mais importantes! ⛰️"
            ],

            'photo_person': [
                "📸 Bela foto! Mas preferimos ver os 'retratos' que o coração desenha no ECG! 💝",
                "😊 Sorrisos fazem bem ao coração, ECGs nos mostram como ele responde! 💓",
                "👥 Pessoas são especiais, mas os batimentos cardíacos são únicos! 💗"
            ]
        }

        responses = humor_responses.get(category, [])
        if responses:
            import random
            return random.choice(responses)

        return None


contextual_response_generator = ContextualResponseGenerator()
